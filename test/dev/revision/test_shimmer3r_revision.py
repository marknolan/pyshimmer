# pyshimmer - API for Shimmer sensor devices
# Copyright (C) 2026  Lukas Magel

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
from __future__ import annotations

import pytest

from pyshimmer import Shimmer3RRevision, EChannelType
from pyshimmer.dev.channels import ESensorGroup


class TestShimmer3RRevision:

    @pytest.fixture
    def revision(self) -> Shimmer3RRevision:
        return Shimmer3RRevision()

    @pytest.mark.parametrize("sensor", Shimmer3RRevision.SENSOR_CHANNEL_ASSIGNMENT)
    def test_sensor_channels_have_dtypes(
        self, revision: Shimmer3RRevision, sensor: ESensorGroup
    ):
        # A channel without a data type cannot be streamed
        for channel in revision.get_enabled_channels([sensor]):
            assert revision.get_channel_dtype(channel) is not None

    def test_accel_hg_dtype(self, revision: Shimmer3RRevision):
        channels = [
            EChannelType.ACCEL_HG_X,
            EChannelType.ACCEL_HG_Y,
            EChannelType.ACCEL_HG_Z,
        ]

        for dtype in revision.get_channel_dtypes(channels):
            assert dtype.size == 2
            assert dtype.signed
            assert dtype.big_endian
            assert dtype.bits == 12

    def test_mag_wr_dtype(self, revision: Shimmer3RRevision):
        channels = [
            EChannelType.MAG_WR_X,
            EChannelType.MAG_WR_Y,
            EChannelType.MAG_WR_Z,
        ]

        for dtype in revision.get_channel_dtypes(channels):
            assert dtype.size == 2
            assert dtype.signed
            assert dtype.little_endian
            assert dtype.bits == 16

    def test_accel_hg_and_mag_wr_sensor_bits(self, revision: Shimmer3RRevision):
        # Bits 6 and 5 of the third byte of the sensor bitfield
        assert revision.sensors2bitfield([ESensorGroup.ACCEL_HG]) == 0x400000
        assert revision.sensors2bitfield([ESensorGroup.MAG_WR]) == 0x200000

        r = revision.deserialize_sensorlist(b"\x00\x00\x60")
        assert r == [ESensorGroup.ACCEL_HG, ESensorGroup.MAG_WR]
