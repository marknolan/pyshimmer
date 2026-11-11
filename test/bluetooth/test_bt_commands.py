# pyshimmer - API for Shimmer sensor devices
# Copyright (C) 2020  Lukas Magel

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

from pyshimmer.bluetooth.bt_commands import (
    GetShimmerHardwareVersion,
    ShimmerCommand,
    GetSamplingRateCommand,
    GetBatteryCommand,
    GetConfigTimeCommand,
    SetConfigTimeCommand,
    GetRealTimeClockCommand,
    SetRealTimeClockCommand,
    GetStatusCommand,
    GetFirmwareVersionCommand,
    Shimmer3InquiryCommand,
    Shimmer3RInquiryCommand,
    StartStreamingCommand,
    StopStreamingCommand,
    StartLoggingCommand,
    StopLoggingCommand,
    GetEXGRegsCommand,
    SetEXGRegsCommand,
    GetExperimentIDCommand,
    SetExperimentIDCommand,
    GetDeviceNameCommand,
    SetDeviceNameCommand,
    DummyCommand,
    DataPacket,
    ResponseCommand,
    SetStatusAckCommand,
    SetSensorsCommand,
    SetSamplingRateCommand,
    GetAllCalibrationCommand,
)
from pyshimmer.bluetooth.bt_serial import BluetoothSerial
from pyshimmer.dev.channels import EChannelType, ESensorGroup
from pyshimmer.dev.fw_version import FirmwareType

from pyshimmer.dev.revisions import HardwareVersion, HardwareRevision, RevisionRegistry
from pyshimmer.test_util import MockSerial


class TestBluetoothCommands:

    @staticmethod
    def create_mock() -> tuple[BluetoothSerial, MockSerial]:
        mock = MockSerial()
        # noinspection PyTypeChecker
        serial = BluetoothSerial(mock)
        return serial, mock

    def assert_cmd(
        self,
        cmd: ShimmerCommand,
        req_data: bytes,
        resp_code: bytes = None,
        resp_data: bytes = None,
        exp_result: any = None,
    ) -> any:
        serial, mock = self.create_mock()

        cmd.send(serial)
        actual_req_data = mock.test_get_write_data()
        assert actual_req_data == req_data

        if resp_code is None:
            assert not cmd.has_response()
            return None

        assert cmd.has_response()
        assert cmd.get_response_code() == resp_code

        mock.test_put_read_data(resp_data)

        act_result = cmd.receive(serial)
        if exp_result is not None:
            assert act_result == exp_result
        return act_result

    def test_response_command_code_conversion(self):
        class TestCommand(ResponseCommand):
            def __init__(self, rcode: int | bytes | tuple[int, ...]):
                super().__init__(RevisionRegistry.REV_SHIMMER3, rcode)

            def send(self, ser: BluetoothSerial) -> None:
                pass

        cmd = TestCommand(10)
        assert cmd.get_response_code() == b"\x0a"

        cmd = TestCommand(20)
        assert cmd.get_response_code() == b"\x14"

        cmd = TestCommand((10,))
        assert cmd.get_response_code() == b"\x0a"

        cmd = TestCommand((10, 20))
        assert cmd.get_response_code() == b"\x0a\x14"

        cmd = TestCommand(b"\x10")
        assert cmd.get_response_code() == b"\x10"

        cmd = TestCommand(b"\x10\x20")
        assert cmd.get_response_code() == b"\x10\x20"

    @pytest.mark.parametrize("rev", RevisionRegistry.ALL_REVISIONS)
    def test_get_sampling_rate_command(self, rev):
        cmd = GetSamplingRateCommand(rev)
        self.assert_cmd(cmd, b"\x03", b"\x04", b"\x04\x40\x00", 512.0)

    @pytest.mark.parametrize("rev", RevisionRegistry.ALL_REVISIONS)
    def test_set_sampling_rate_command(self, rev: HardwareRevision):
        cmd = SetSamplingRateCommand(rev, sr=512.0)
        self.assert_cmd(cmd, b"\x05\x40\x00")

    @pytest.mark.parametrize("rev", RevisionRegistry.ALL_REVISIONS)
    def test_get_battery_state_command(self, rev: HardwareRevision):
        cmd = GetBatteryCommand(rev, in_percent=True)
        self.assert_cmd(cmd, b"\x95", b"\x8a\x94", b"\x8a\x94\x30\x0b\x80", 100)

        cmd = GetBatteryCommand(rev, in_percent=False)
        self.assert_cmd(
            cmd, b"\x95", b"\x8a\x94", b"\x8a\x94\x2e\x0b\x80", 4.168246153846154
        )

    @pytest.mark.parametrize("rev", RevisionRegistry.ALL_REVISIONS)
    def test_set_sensors_command(self, rev: HardwareRevision):
        sensors = [
            ESensorGroup.GYRO,
            ESensorGroup.INT_CH_A1,
            ESensorGroup.PRESSURE,
        ]
        cmd = SetSensorsCommand(rev, sensors)
        self.assert_cmd(cmd, b"\x08\x40\x01\x04")

    @pytest.mark.parametrize("rev", RevisionRegistry.ALL_REVISIONS)
    def test_get_config_time_command(self, rev: HardwareRevision):
        cmd = GetConfigTimeCommand(rev)
        self.assert_cmd(cmd, b"\x87", b"\x86", b"\x86\x02\x34\x32", 42)

    @pytest.mark.parametrize("rev", RevisionRegistry.ALL_REVISIONS)
    def test_set_config_time_command(self, rev: HardwareRevision):
        cmd = SetConfigTimeCommand(rev, 43)
        self.assert_cmd(cmd, b"\x85\x02\x34\x33")

    @pytest.mark.parametrize("rev", RevisionRegistry.ALL_REVISIONS)
    def test_get_rtc(self, rev: HardwareRevision):
        cmd = GetRealTimeClockCommand(rev)
        r = self.assert_cmd(
            cmd, b"\x91", b"\x90", b"\x90\x1f\xb1\x93\x09\x00\x00\x00\x00"
        )
        assert r == pytest.approx(4903.3837585)

    @pytest.mark.parametrize("rev", RevisionRegistry.ALL_REVISIONS)
    def test_set_rtc(self, rev: HardwareRevision):
        cmd = SetRealTimeClockCommand(rev, 10)
        self.assert_cmd(cmd, b"\x8f\x00\x00\x05\x00\x00\x00\x00\x00")

    @pytest.mark.parametrize("rev", RevisionRegistry.ALL_REVISIONS)
    def test_get_status_command(self, rev: HardwareRevision):
        cmd = GetStatusCommand(rev)
        expected_result = [True, False, True, False, False, True, False, False]
        self.assert_cmd(cmd, b"\x72", b"\x8a\x71", b"\x8a\x71\x25", expected_result)

    @pytest.mark.parametrize("rev", RevisionRegistry.ALL_REVISIONS)
    def test_get_firmware_version_command(self, rev: HardwareRevision):
        cmd = GetFirmwareVersionCommand(rev)
        fw_type, major, minor, patch = self.assert_cmd(
            cmd, b"\x2e", b"\x2f", b"\x2f\x03\x00\x00\x00\x0b\x00"
        )
        assert fw_type == FirmwareType.LogAndStream
        assert major == 0
        assert minor == 11
        assert patch == 0

    def test_inquiry_command_shimmer3(self):
        cmd = Shimmer3InquiryCommand(RevisionRegistry.REV_SHIMMER3)
        sr, buf_size, ctypes = self.assert_cmd(
            cmd, b"\x01", b"\x02", b"\x02\x40\x00\x01\xff\x01\x09\x01\x01\x12"
        )

        assert sr == 512.0
        assert buf_size == 1
        assert ctypes == [EChannelType.INTERNAL_ADC_A1]

    def test_inquiry_command_shimmer3r(self):
        cmd = Shimmer3RInquiryCommand(RevisionRegistry.REV_SHIMMER3R)
        sr, buf_size, ctypes = self.assert_cmd(
            cmd,
            b"\x01",
            b"\x02",
            b"\x02\x40\x00\x02\x00\x01\x09\x00\x00\x00\x01\x01\x12",
        )

        assert sr == 512.0
        assert buf_size == 1
        assert ctypes == [EChannelType.INTERNAL_ADC_A1]

    @pytest.mark.parametrize("rev", RevisionRegistry.ALL_REVISIONS)
    def test_start_streaming_command(self, rev: HardwareRevision):
        cmd = StartStreamingCommand(rev)
        self.assert_cmd(cmd, b"\x07")

    @pytest.mark.parametrize("rev", RevisionRegistry.ALL_REVISIONS)
    def test_stop_streaming_command(self, rev: HardwareRevision):
        cmd = StopStreamingCommand(rev)
        self.assert_cmd(cmd, b"\x20")

    @pytest.mark.parametrize("rev", RevisionRegistry.ALL_REVISIONS)
    def test_start_logging_command(self, rev: HardwareRevision):
        cmd = StartLoggingCommand(rev)
        self.assert_cmd(cmd, b"\x92")

    @pytest.mark.parametrize("rev", RevisionRegistry.ALL_REVISIONS)
    def test_stop_logging_command(self, rev: HardwareRevision):
        cmd = StopLoggingCommand(rev)
        self.assert_cmd(cmd, b"\x93")

    @pytest.mark.parametrize("rev", RevisionRegistry.ALL_REVISIONS)
    def test_get_exg_register_command(self, rev: HardwareRevision):
        cmd = GetEXGRegsCommand(rev, 1)
        r = self.assert_cmd(
            cmd,
            b"\x63\x01\x00\x0a",
            b"\x62",
            b"\x62\x0a\x00\x80\x10\x00\x00\x00\x00\x00\x02\x01",
        )
        assert r.binary == b"\x00\x80\x10\x00\x00\x00\x00\x00\x02\x01"

    @pytest.mark.parametrize("rev", RevisionRegistry.ALL_REVISIONS)
    def test_get_exg_reg_fail(self, rev: HardwareRevision):
        serial, mock = self.create_mock()
        cmd = GetEXGRegsCommand(rev, 1)

        mock.test_put_read_data(b"\x62\x04\x01\x02\x03\x04")
        with pytest.raises(ValueError):
            cmd.receive(serial)

    @pytest.mark.parametrize("rev", RevisionRegistry.ALL_REVISIONS)
    def test_get_allcalibration_command(self, rev: HardwareRevision):
        response_data = (
            b"\x2d\x08\xcd\x08\xcd\x08\xcd\x00\x5c\x00\x5c\x00\x5c\x00\x9c\x00"
            b"\x9c\x00\x00\x00\x00\x9c\x00\x00\x00\x00\x00\x00\x19\x96\x19\x96"
            b"\x19\x96\x00\x9c\x00\x9c\x00\x00\x00\x00\x9c\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x06\x87\x06\x87\x06\x87\x00\x9c\x00\x64"
            b"\x00\x00\x00\x00\x9c"
        )
        expected_result = (
            b"\x08\xcd\x08\xcd\x08\xcd\x00\x5c\x00\x5c\x00\x5c\x00\x9c\x00\x9c"
            b"\x00\x00\x00\x00\x9c\x00\x00\x00\x00\x00\x00\x19\x96\x19\x96\x19"
            b"\x96\x00\x9c\x00\x9c\x00\x00\x00\x00\x9c\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x06\x87\x06\x87\x06\x87\x00\x9c\x00\x64\x00"
            b"\x00\x00\x00\x9c"
        )

        cmd = GetAllCalibrationCommand(rev)
        r = self.assert_cmd(cmd, b"\x2c", b"\x2d", response_data)
        assert r.binary == expected_result

    @pytest.mark.parametrize("rev", RevisionRegistry.ALL_REVISIONS)
    def test_set_exg_register_command(self, rev: HardwareRevision):
        cmd = SetEXGRegsCommand(rev, 1, 0x02, b"\x10\x00")
        self.assert_cmd(cmd, b"\x61\x01\x02\x02\x10\x00")

    @pytest.mark.parametrize("rev", RevisionRegistry.ALL_REVISIONS)
    def test_get_experiment_id_command(self, rev: HardwareRevision):
        cmd = GetExperimentIDCommand(rev)
        self.assert_cmd(cmd, b"\x7e", b"\x7d", b"\x7d\x06a_test", "a_test")

    @pytest.mark.parametrize("rev", RevisionRegistry.ALL_REVISIONS)
    def test_set_experiment_id_command(self, rev: HardwareRevision):
        cmd = SetExperimentIDCommand(rev, "A_Test")
        self.assert_cmd(cmd, b"\x7c\x06A_Test")

    @pytest.mark.parametrize("rev", RevisionRegistry.ALL_REVISIONS)
    def test_get_device_name_command(self, rev: HardwareRevision):
        cmd = GetDeviceNameCommand(rev)
        self.assert_cmd(cmd, b"\x7b", b"\x7a", b"\x7a\x05S_PPG", "S_PPG")

    @pytest.mark.parametrize("rev", RevisionRegistry.ALL_REVISIONS)
    def test_get_hardware_version(self, rev: HardwareRevision):
        cmd = GetShimmerHardwareVersion(rev)
        self.assert_cmd(cmd, b"\x3f", b"\x25", b"\x25\x00", HardwareVersion.SHIMMER1)
        self.assert_cmd(cmd, b"\x3f", b"\x25", b"\x25\x01", HardwareVersion.SHIMMER2)
        self.assert_cmd(cmd, b"\x3f", b"\x25", b"\x25\x02", HardwareVersion.SHIMMER2R)
        self.assert_cmd(cmd, b"\x3f", b"\x25", b"\x25\x03", HardwareVersion.SHIMMER3)
        self.assert_cmd(cmd, b"\x3f", b"\x25", b"\x25\x0a", HardwareVersion.SHIMMER3R)
        self.assert_cmd(cmd, b"\x3f", b"\x25", b"\x25\x04", HardwareVersion.UNKNOWN)

    @pytest.mark.parametrize("rev", RevisionRegistry.ALL_REVISIONS)
    def test_set_device_name_command(self, rev: HardwareRevision):
        cmd = SetDeviceNameCommand(rev, "S_PPG")
        self.assert_cmd(cmd, b"\x79\x05S_PPG")

    @pytest.mark.parametrize("rev", RevisionRegistry.ALL_REVISIONS)
    def test_set_status_ack_command(self, rev: HardwareRevision):
        cmd = SetStatusAckCommand(rev, enabled=True)
        self.assert_cmd(cmd, b"\xa3\x01")

        cmd = SetStatusAckCommand(rev, enabled=False)
        self.assert_cmd(cmd, b"\xa3\x00")

    @pytest.mark.parametrize("rev", RevisionRegistry.ALL_REVISIONS)
    def test_dummy_command(self, rev: HardwareRevision):
        cmd = DummyCommand(rev)
        self.assert_cmd(cmd, b"\x96")

    def test_data_packet_shimmer3r_accel_hg_and_mag_wr(self):
        # The first 46 bytes of three packets a Shimmer3R streamed at 1024 Hz with
        # every sensor on. The packets carry three more channels after these.
        rev = RevisionRegistry.REV_SHIMMER3R
        channel_ids = [
            # Magnetometer, gyroscope, low-noise accelerometer
            *(0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x00, 0x01, 0x02),
            # Pressure and temperature
            *(0x1B, 0x1A),
            # High-g accelerometer, wide-range accelerometer, wide-range magnetometer
            *(0x14, 0x15, 0x16, 0x04, 0x05, 0x06, 0x17, 0x18, 0x19),
        ]
        channels = [EChannelType.TIMESTAMP] + [
            EChannelType.enum_for_id(i) for i in channel_ids
        ]
        packets = [
            "005c20a8270051001efdb7ff2700150088f3e83c99ef00925e00678001"
            "50ffb00030ecc408f14cf3000000000000",
            "007c20a8270051001efdafff2d00080078f3d13c83ef00925e00678001"
            "700020ffb0a8c4ecf000f3000000000000",
            "009c20a8270051001efda9ff1f000700a0f3073d91ef008f5e00698001"
            "10ffe0ffe0b8c40cf164f3000000000000",
        ]
        timestamps = [11018332, 11018364, 11018396]
        accel_hg = [(21, -5, 3), (23, 2, -5), (17, -2, -2)]
        accel_wr = [
            (-15124, -3832, -3252),
            (-15192, -3860, -3328),
            (-15176, -3828, -3228),
        ]

        for packet, ts, hg, wr in zip(packets, timestamps, accel_hg, accel_wr):
            serial, mock = self.create_mock()
            pkt = DataPacket(rev, list(zip(channels, rev.get_channel_dtypes(channels))))
            mock.test_put_read_data(bytes.fromhex(packet))
            pkt.receive(serial)

            assert mock.test_get_remaining_read_data() == b""
            assert pkt[EChannelType.TIMESTAMP] == ts
            assert (
                pkt[EChannelType.ACCEL_HG_X],
                pkt[EChannelType.ACCEL_HG_Y],
                pkt[EChannelType.ACCEL_HG_Z],
            ) == hg
            assert (
                pkt[EChannelType.ACCEL_WR_X],
                pkt[EChannelType.ACCEL_WR_Y],
                pkt[EChannelType.ACCEL_WR_Z],
            ) == wr
            # The wide-range magnetometer read zero throughout this recording
            assert (
                pkt[EChannelType.MAG_WR_X],
                pkt[EChannelType.MAG_WR_Y],
                pkt[EChannelType.MAG_WR_Z],
            ) == (0, 0, 0)

    @pytest.mark.parametrize("rev", RevisionRegistry.ALL_REVISIONS)
    def test_data_packet(self, rev: HardwareRevision):
        serial, mock = self.create_mock()

        channels = [EChannelType.TIMESTAMP, EChannelType.INTERNAL_ADC_A1]
        data_types = rev.get_channel_dtypes(channels)
        ch_and_types = list(zip(channels, data_types))

        pkt = DataPacket(rev, ch_and_types)
        assert pkt.channels == channels
        assert pkt.channel_types == data_types

        mock.test_put_read_data(b"\x00\xde\xd0\xb2\x26\x07")
        pkt.receive(serial)

        assert pkt[EChannelType.TIMESTAMP] == 0xB2D0DE
        assert pkt[EChannelType.INTERNAL_ADC_A1] == 0x0726
