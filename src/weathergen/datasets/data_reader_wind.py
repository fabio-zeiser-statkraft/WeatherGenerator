# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from pathlib import Path
from typing import override

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from weathergen.datasets.data_reader_base import (
    DataReaderTimestep,
    ReaderData,
    TimeWindowHandler,
    TIndex,
    check_reader_data,
)

_logger = logging.getLogger(__name__)


class DataReaderWind(DataReaderTimestep):
    "Wrapper for WindPower datasets from Statkraft in Zarr"

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        filename: Path,
        stream_info: dict,
    ) -> None:
        """
        Construct data reader for the dataset

        Parameters
        ----------
        filename :
            filename (and path) of dataset
        stream_info :
            information about stream

        Returns
        -------
        None
        """

        # open  dataset to peak that it is compatible with requested parameters
        ds = xr.open_zarr(filename).production

        # If there is no overlap with the time range, the dataset will be empty
        if tw_handler.t_start >= ds.date.max() or tw_handler.t_end <= ds.date.min():
            name = stream_info["name"]
            _logger.warning(
                f"{name} is not supported over data loader window. Stream is skipped."
            )
            super().__init__(tw_handler, stream_info)
            self.init_empty()
            return

        kwargs = {}
        if "frequency" in stream_info:
            # kwargs["frequency"] = str_to_timedelta(stream_info["frequency"])
            assert False, "Frequency sub-sampling currently not supported"

        period = (ds.date[1] - ds.date[0]).values
        data_start_time = ds.date[0].values
        data_end_time = ds.date[-1].values
        assert data_start_time is not None and data_end_time is not None, (
            data_start_time,
            data_end_time,
        )
        super().__init__(
            tw_handler,
            stream_info,
            data_start_time,
            data_end_time,
            period,
        )
        # If there is no overlap with the time range, no need to keep the dataset.
        if tw_handler.t_start >= data_end_time or tw_handler.t_end <= data_start_time:
            self.init_empty()
            return
        else:
            self.ds = ds
            self.len = len(ds)

        # self.offset_data_channels = 4
        # self.fillvalue = ds["air_temperature"][0, 0].values.item()
        # self.fill_value_per_channel = -100 # signals a impossible value

        # caches lats and lons
        # TODO: Currently the lat / lon is not provided. We hack a value
        _logger.warning(f"Assigning arbitrary lat/lon for wind power. WIP")

        # ds["latitude"] = np.arange(len(ds["variable"]))
        # ds["longitude"] = np.arange(len(ds["variable"]))
        ds = ds.assign_coords(latitude=("variable", np.arange(len(ds["variable"]))))
        ds = ds.assign_coords(longitude=("variable", np.arange(len(ds["variable"]))))

        self.latitudes = _clip_lat(np.array(ds.latitude, dtype=np.float32))
        self.longitudes = _clip_lon(np.array(ds.longitude, dtype=np.float32))

        # self.latitudes = _clip_lat(np.array([0], dtype=np.float32))
        # self.longitudes = _clip_lon(np.array([0], dtype=np.float32))

        self.geoinfos = []  # np.array(ds.altitude, dtype=np.float32)
        self.geoinfo_channels = []  # ["altitude"]
        self.geoinfo_idx = []  # [2]

        # self.channels_file = [k for k in self.ds.keys()]

        # select/filter requested source channels
        self.source_idx = []  # self.select_channels(ds, "source")
        self.source_channels = []  # [self.channels_file[i] for i in self.source_idx]

        # select/filter requested target channels

        # self.target_idx = np.arange(len(self.ds["variable"]))#self.select_channels(ds, "target")
        # self.target_channels = self.ds["variable"].values.tolist()

        # TODO: actually, the target here is always included, and it is the wind power production
        #       Another selector could be what sites to include / exclude
        self.target_idx = np.array([0])
        self.target_channels = ["wind_power_in_mpa"]
        # But how do you deal with different mean/std for each geo-location?

        ds_name = stream_info["name"]
        _logger.info(f"{ds_name}: source channels: {self.source_channels}")
        _logger.info(f"{ds_name}: target channels: {self.target_channels}")
        _logger.info(f"{ds_name}: geoinfo channels: {self.geoinfo_channels}")

        self.properties = {
            "stream_id": 0,
        }

        self.mean, self.stdev = self.compute_mean_stdev()

    def compute_mean_stdev(self) -> tuple[np.array, np.array]:
        _logger.info("Starting computation of mean and stdev.")

        # calculate for all locations (so best to train on capacity normalized values)
        # Alternative to capacity-normalization is location weighting.
        mean = np.atleast_1d(self.ds.mean().compute().values)
        stdev = np.atleast_1d(self.ds.std().compute().values)

        _logger.info("Finished computation of mean and stdev.")

        return mean, stdev

    @override
    def init_empty(self) -> None:
        super().init_empty()
        self.ds = None
        self.len = 0

    @override
    def length(self) -> int:
        return self.len

    @override
    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        """
        Get data for window (for either source or target, through public interface)

        Parameters
        ----------
        idx : int
            Index of temporal window
        channels_idx : np.array
            Selection of channels

        Returns
        -------
        ReaderData providing coords, geoinfos, data, datetimes
        """

        (t_idxs, dtr) = self._get_dataset_idxs(idx)

        if (
            self.ds is None
            or self.len == 0
            or len(t_idxs) == 0
            or len(channels_idx) == 0
        ):
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
            )

        assert t_idxs[0] >= 0, "index must be non-negative"
        didx_start = t_idxs[0]
        # End is inclusive
        didx_end = t_idxs[-1] + 1

        if channels_idx != [0]:
            raise NotImplementedError(
                "Only channel 0 (wind power) is supported currently"
            )

        # extract number of time steps and collapse ensemble dimension
        # ds is a wrapper around zarr with get_coordinate_selection not being exposed since
        # subsetting is pushed to the ctor via frequency argument; this also ensures that no sub-
        # sampling is required here

        # TODO: Rename "variable" to "market price area" or similar
        data = self.ds.isel(date=slice(didx_start, didx_end))

        data = data.transpose("date", "variable")
        data = data.stack(z=("date", "variable")).values
        data = np.atleast_2d(data).transpose()

        # dims
        # channel, time-window, location
        # time-window*location , channel

        # construct lat/lon coords
        latlon = np.concatenate(
            [
                np.expand_dims(self.latitudes, 0),
                np.expand_dims(self.longitudes, 0),
            ],
            axis=0,
        ).transpose()
        # repeat latlon len(t_idxs) times
        coords = np.vstack((latlon,) * len(t_idxs))

        # empty geoinfos for anemoi
        # TODO: altitudes
        geoinfos = np.zeros((len(data), 0), dtype=data.dtype)

        # date time matching #data points of data
        # Assuming a fixed frequency for the dataset
        datetimes = np.repeat(
            self.ds.date[didx_start:didx_end].values, len(data) // len(t_idxs)
        )

        rd = ReaderData(
            coords=coords,
            geoinfos=geoinfos,
            data=data,
            datetimes=datetimes,
        )

        check_reader_data(rd, dtr)
        return rd

    # def select_channels(self, ds, ch_type: str) -> NDArray[np.int64]:
    #     """
    #     Select source or target channels

    #     Parameters
    #     ----------
    #     ds0 :
    #         raw anemoi dataset with available channels
    #     ch_type :
    #         "source" or "target", i.e channel type to select

    #     Returns
    #     -------
    #     ReaderData providing coords, geoinfos, data, datetimes

    #     """

    #     channels_file = [k for k in ds.keys()][self.offset_data_channels :]

    #     channels = self.stream_info.get(ch_type, channels_file)
    #     channels_exclude = self.stream_info.get(ch_type + "_exclude", [])
    #     # sanity check
    #     is_empty = len(channels) == 0 if channels is not None else False
    #     if is_empty:
    #         stream_name = self.stream_info["name"]
    #         _logger.warning(f"No channel for {stream_name} for {ch_type}.")

    #     chs_idx = np.sort([channels_file.index(ch) for ch in channels])
    #     chs_idx_exclude = np.sort([channels_file.index(ch) for ch in channels_exclude])
    #     chs_idx = [idx for idx in chs_idx if idx not in chs_idx_exclude]

    #     return np.array(chs_idx) + self.offset_data_channels


def _clip_lat(lats: NDArray) -> NDArray[np.float32]:
    """
    Clip latitudes to the range [-90, 90] and ensure periodicity.
    """
    return (2 * np.clip(lats, -90.0, 90.0) - lats).astype(np.float32)


def _clip_lon(lons: NDArray) -> NDArray[np.float32]:
    """
    Clip longitudes to the range [-180, 180] and ensure periodicity.
    """
    return ((lons + 180.0) % 360.0 - 180.0).astype(np.float32)


if __name__ == "__main__":
    # Convert string dates to np.datetime64
    t_start = np.datetime64("2021")
    t_end = np.datetime64("2022")
    reader = DataReaderWind(
        tw_handler=TimeWindowHandler(t_start, t_end, 6, 1),
        filename=Path(
            "/p/project1/weatherai/zeiser2/data/wind_observations_wg/wind_onshore.zarr/"
        ),
        stream_info={"name": "wind_onshore"},
    )
    # data = reader._get(0, list(range(10)))
    # data = reader._get(0, reader.target_idx)
    data = reader._get(0, [0])
    print(data.coords.shape)
    print(data.data.shape)
    print(reader.mean.shape)
    print(reader.stdev.shape)
    print(reader.target_idx.shape)
