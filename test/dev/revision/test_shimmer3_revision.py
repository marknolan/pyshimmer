# pyshimmer - API for Shimmer sensor devices
# Copyright (C) 2025  Lukas Magel

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

import io
import itertools
import struct

import numpy as np
import pytest

from pyshimmer import Shimmer3Revision, EChannelType
from pyshimmer.bluetooth.bt_commands import DataPacket
from pyshimmer.bluetooth.bt_serial import BluetoothSerial
from pyshimmer.dev.channels import ESensorGroup


class _ByteStream(io.BytesIO):
    # A blocking serial without a timeout, as pyshimmer expects; a short read past the
    # end of the bytes aborts the read
    timeout = None


class TestShimmer3Revision:

    @pytest.fixture
    def revision(self) -> Shimmer3Revision:
        return Shimmer3Revision()

    def test_sr2dr(self, revision: Shimmer3Revision):
        r = revision.sr2dr(1024.0)
        assert r == 32

        r = revision.sr2dr(500.0)
        assert r == 66

    def test_dr2sr(self, revision: Shimmer3Revision):
        r = revision.dr2sr(65)
        assert r == pytest.approx(504, abs=0.5)

        r = revision.dr2sr(32)
        assert r == 1024.0

        r = revision.dr2sr(64)
        assert r == 512.0

    def test_sec2ticks(self, revision: Shimmer3Revision):
        r = revision.sec2ticks(1.0)
        assert r == 32768

        r = revision.sec2ticks(np.array([1.0, 2.0]))
        np.testing.assert_array_equal(r, np.array([32768, 65536]))

    def test_ticks2sec(self, revision: Shimmer3Revision):
        r = revision.ticks2sec(32768)
        assert r == 1.0

        r = revision.ticks2sec(65536)
        assert r == 2.0

        i = np.array([32768, 65536])
        r = revision.ticks2sec(i)
        np.testing.assert_array_equal(r, np.array([1.0, 2.0]))

    def test_get_channel_dtypes(self, revision: Shimmer3Revision):
        r = revision.get_channel_dtypes([])
        assert r == []

        r = revision.get_channel_dtypes(())
        assert r == []

        r = revision.get_channel_dtypes(
            [EChannelType.INTERNAL_ADC_A0, EChannelType.INTERNAL_ADC_A0]
        )
        assert len(r) == 2
        assert r[0] == r[1]

        channels = [EChannelType.INTERNAL_ADC_A1, EChannelType.GYRO_Y]
        r = revision.get_channel_dtypes(channels)

        assert len(r) == 2
        first, second = r

        assert first.size == 2
        assert first.little_endian is True
        assert first.signed is False

        assert second.size == 2
        assert second.little_endian is False
        assert second.signed is True

    def test_channel_dtype_assignment(self, revision: Shimmer3Revision):
        for channel in EChannelType:
            r = revision.get_channel_dtypes([channel])
            assert len(r) > 0

    def test_get_enabled_channels(self, revision: Shimmer3Revision):
        r = revision.get_enabled_channels([])
        assert r == []

        r = revision.get_enabled_channels(())
        assert r == []

        r = revision.get_enabled_channels(
            [ESensorGroup.PRESSURE, ESensorGroup.ACCEL_LN]
        )

        assert r == [
            EChannelType.TEMPERATURE,
            EChannelType.PRESSURE,
            EChannelType.ACCEL_LN_X,
            EChannelType.ACCEL_LN_Y,
            EChannelType.ACCEL_LN_Z,
        ]

    def test_sensor_group_assignment(self, revision: Shimmer3Revision):
        for group in ESensorGroup:
            r = revision.get_enabled_channels([group])

            if group != ESensorGroup.TEMP:
                assert len(r) > 0

    def test_sensor_list_to_bitfield(self, revision: Shimmer3Revision):
        r = revision.sensors2bitfield((ESensorGroup.ACCEL_LN, ESensorGroup.EXT_CH_A1))
        assert r == 0x81

        r = revision.sensors2bitfield((ESensorGroup.STRAIN, ESensorGroup.INT_CH_A1))
        assert r == 0x8100

        r = revision.sensors2bitfield((ESensorGroup.INT_CH_A2, ESensorGroup.TEMP))
        assert r == 0x820000

    def test_bitfield_to_sensors(self, revision: Shimmer3Revision):
        r = revision.bitfield2sensors(0x81)
        assert r == [ESensorGroup.ACCEL_LN, ESensorGroup.EXT_CH_A1]

        r = revision.bitfield2sensors(0x8100)
        assert r == [ESensorGroup.INT_CH_A1, ESensorGroup.STRAIN]

        r = revision.bitfield2sensors(0x820000)
        assert r == [
            ESensorGroup.INT_CH_A2,
            ESensorGroup.TEMP,
        ]

    def test_sensor_bit_assignment_uniqueness(self, revision: Shimmer3Revision):
        for group1, group2 in itertools.product(ESensorGroup, ESensorGroup):
            if group1 == group2:
                continue

            bitfield1 = revision.sensors2bitfield([group1])
            bitfield2 = revision.sensors2bitfield([group2])
            assert bitfield1 != bitfield2

    def test_serialize_sensorlist(self, revision: Shimmer3Revision):
        r = revision.serialize_sensorlist([])
        assert r == b"\x00\x00\x00"

        r = revision.serialize_sensorlist([ESensorGroup.GSR, ESensorGroup.BATTERY])
        assert r == b"\x04\x20\x00"

    def test_deserialize_sensorlist(self, revision: Shimmer3Revision):
        r = revision.deserialize_sensorlist(b"\x00\x00\x00")
        assert r == []

        r = revision.deserialize_sensorlist(b"\x01\x80\x01")
        assert r == [
            ESensorGroup.EXT_CH_A1,
            ESensorGroup.STRAIN,
        ]

    def test_serialize_deserialize(self, revision: Shimmer3Revision):
        for group in ESensorGroup:
            bitfield = revision.serialize_sensorlist([group])
            group_deserialized = revision.deserialize_sensorlist(bitfield)
            assert [group] == group_deserialized

    def test_sort_sensors(self, revision: Shimmer3Revision):
        sensors = [ESensorGroup.BATTERY, ESensorGroup.ACCEL_LN]
        expected = [ESensorGroup.ACCEL_LN, ESensorGroup.BATTERY]
        r = revision.sort_sensors(sensors)
        assert r == expected

        sensors = [
            ESensorGroup.EXT_CH_A2,
            ESensorGroup.MAG_WR,
            ESensorGroup.ACCEL_LN,
            ESensorGroup.EXT_CH_A2,
        ]
        expected = [
            ESensorGroup.ACCEL_LN,
            ESensorGroup.EXT_CH_A2,
            ESensorGroup.EXT_CH_A2,
            ESensorGroup.MAG_WR,
        ]
        r = revision.sort_sensors(sensors)
        assert r == expected

    def test_unwrap_device_timestamps(self, revision: Shimmer3Revision):
        ts_wrapped = np.array([0, 1, 2, 2**24 - 1, 0, 2**24])
        expected = np.array([0, 1, 2, 2**24 - 1, 2**24, 2 * 2**24])
        actual = revision.unwrap_device_timestamps(ts_wrapped)
        np.testing.assert_equal(actual, expected)

        ts_wrapped = np.array([0, 10, 20, 30, 5, 15, 25, 35])
        expected = np.array(
            [0, 10, 20, 30, 5 + 2**24, 15 + 2**24, 25 + 2**24, 35 + 2**24]
        )
        actual = revision.unwrap_device_timestamps(ts_wrapped)
        np.testing.assert_equal(actual, expected)

    def test_get_exg_chip_and_channel(self, revision: Shimmer3Revision):
        assert revision.get_exg_chip_and_channel(EChannelType.EXG1_CH1_16BIT) == (0, 0)
        assert revision.get_exg_chip_and_channel(EChannelType.EXG1_CH2_16BIT) == (0, 1)
        assert revision.get_exg_chip_and_channel(EChannelType.EXG2_CH1_16BIT) == (1, 0)
        assert revision.get_exg_chip_and_channel(EChannelType.EXG2_CH2_16BIT) == (1, 1)
        assert revision.get_exg_chip_and_channel(EChannelType.EXG1_CH1_24BIT) == (0, 0)
        assert revision.get_exg_chip_and_channel(EChannelType.EXG1_CH2_24BIT) == (0, 1)
        assert revision.get_exg_chip_and_channel(EChannelType.EXG2_CH1_24BIT) == (1, 0)
        assert revision.get_exg_chip_and_channel(EChannelType.EXG2_CH2_24BIT) == (1, 1)

        with pytest.raises(ValueError):
            revision.get_exg_chip_and_channel(EChannelType.ACCEL_HG_X)

    @pytest.mark.parametrize("sensor", [ESensorGroup.ACCEL_HG, ESensorGroup.MAG_WR])
    def test_alt_accel_and_mag_channels_have_dtypes(
        self, revision: Shimmer3Revision, sensor: ESensorGroup
    ):
        # The firmware sends two bytes for each of these channels when the sensor is on,
        # so a channel without a data type breaks every stream it is in
        channels = revision.get_enabled_channels([sensor])
        assert len(channels) == 3
        for dtype in revision.get_channel_dtypes(channels):
            assert dtype is not None
            assert dtype.size == 2
            assert dtype.signed

    def test_alt_accel_dtype(self, revision: Shimmer3Revision):
        # The MPU-9150/9250 or ICM-20948 accelerometer, read from ACCEL_XOUT_H
        channels = [
            EChannelType.ACCEL_HG_X,
            EChannelType.ACCEL_HG_Y,
            EChannelType.ACCEL_HG_Z,
        ]
        for dtype in revision.get_channel_dtypes(channels):
            assert dtype.big_endian

    def test_alt_mag_dtype(self, revision: Shimmer3Revision):
        # The AK8975/AK8963 or AK09916 magnetometer, read from HXL
        channels = [
            EChannelType.MAG_WR_X,
            EChannelType.MAG_WR_Y,
            EChannelType.MAG_WR_Z,
        ]
        for dtype in revision.get_channel_dtypes(channels):
            assert dtype.little_endian

    def test_alt_accel_and_mag_sensor_bits(self, revision: Shimmer3Revision):
        assert revision.sensors2bitfield([ESensorGroup.ACCEL_HG]) == 0x400000
        assert revision.sensors2bitfield([ESensorGroup.MAG_WR]) == 0x200000

    def test_data_packet_with_alt_accel_and_mag(self, revision: Shimmer3Revision):
        # A packet laid out as the Shimmer3 firmware packs it on an ICM-20948 board:
        # low-noise accel, battery, gyro, wide-range accel, mag, then the alternative
        # accel and mag. On that board the wide-range accel is the same chip reading as
        # the alternative accel, swapped to little-endian with Y negated, and the mag the
        # same reading as the alternative mag with Z negated, so each pair checks the
        # other's byte order.
        accel = (1234, -1234, 16384)
        mag = (291, -240, 512)
        channels = [
            EChannelType.TIMESTAMP,
            EChannelType.ACCEL_LN_X,
            EChannelType.ACCEL_LN_Y,
            EChannelType.ACCEL_LN_Z,
            EChannelType.VBATT,
            EChannelType.GYRO_X,
            EChannelType.GYRO_Y,
            EChannelType.GYRO_Z,
            EChannelType.ACCEL_WR_X,
            EChannelType.ACCEL_WR_Y,
            EChannelType.ACCEL_WR_Z,
            EChannelType.MAG_REG_X,
            EChannelType.MAG_REG_Y,
            EChannelType.MAG_REG_Z,
            EChannelType.ACCEL_HG_X,
            EChannelType.ACCEL_HG_Y,
            EChannelType.ACCEL_HG_Z,
            EChannelType.MAG_WR_X,
            EChannelType.MAG_WR_Y,
            EChannelType.MAG_WR_Z,
        ]
        body = (
            b"\x10\x20\x30"
            + struct.pack("<3h", 2048, 2047, 2049)
            + struct.pack("<h", 2700)
            + struct.pack(">3h", 100, -100, 0)
            + struct.pack("<3h", accel[0], -accel[1], accel[2])
            + struct.pack("<3h", mag[0], mag[1], -mag[2])
            + struct.pack(">3h", *accel)
            + struct.pack("<3h", *mag)
        )
        assert len(body) == 3 + 38

        # Two packets back to back: the second must start where the first one ends
        stream = b"\x00" + body + b"\x00" + body
        ser = BluetoothSerial(_ByteStream(stream))
        pkt = DataPacket(
            revision, list(zip(channels, revision.get_channel_dtypes(channels)))
        )
        for _ in range(2):
            pkt.receive(ser)

            assert pkt[EChannelType.TIMESTAMP] == 0x302010
            assert pkt[EChannelType.ACCEL_HG_X] == accel[0]
            assert pkt[EChannelType.ACCEL_HG_Y] == accel[1]
            assert pkt[EChannelType.ACCEL_HG_Z] == accel[2]
            assert pkt[EChannelType.MAG_WR_X] == mag[0]
            assert pkt[EChannelType.MAG_WR_Y] == mag[1]
            assert pkt[EChannelType.MAG_WR_Z] == mag[2]

            assert pkt[EChannelType.ACCEL_WR_X] == pkt[EChannelType.ACCEL_HG_X]
            assert pkt[EChannelType.ACCEL_WR_Y] == -pkt[EChannelType.ACCEL_HG_Y]
            assert pkt[EChannelType.ACCEL_WR_Z] == pkt[EChannelType.ACCEL_HG_Z]
            assert pkt[EChannelType.MAG_REG_X] == pkt[EChannelType.MAG_WR_X]
            assert pkt[EChannelType.MAG_REG_Z] == -pkt[EChannelType.MAG_WR_Z]
            assert pkt[EChannelType.GYRO_Y] == -100
