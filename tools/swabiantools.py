import TimeTagger as _TimeTagger
import numpy as _np
from collections.abc import Sequence as _Sequence
from enum import Enum
from dataclasses import dataclass as _dataclass


class SignalEdgeEnum(Enum):
    RISEEDGE = 1
    FALLEDGE = 2


@_dataclass
class SignalData:
    name: str
    sense: SignalEdgeEnum
    trigger_voltage: float


class SignalTypeEnum(Enum):
    """Enumeration of signal Types."""
    NIM = SignalData("NIM", SignalEdgeEnum.FALLEDGE, -.5)
    TTL = SignalData("TTL", SignalEdgeEnum.RISEEDGE, 2.5)


def _get_delay_and_jitter(x, y):
    # Helper method to calculate the mean time difference of a histogram and the standard deviation.
    mean = _np.average(x, weights=y)
    std = _np.sqrt(_np.average((x - mean) ** 2, weights=y))
    return mean, std


def calibrate_period(
    tagger: _TimeTagger.TimeTagger,
    channel_number: int,
    signal_type: SignalTypeEnum = SignalTypeEnum.NIM,
    test_time: int = int(1e12),
    return_data: bool = False,
    debug=False,
):
    """Calibrate trigger channel period.

    Parameters
    ==========
        tagger: TimeTagger
            ...
        channel_number: int
            channel number, positive
        signal_type: SignalTypeEnum, default = SignalTypeEnum.NIM
            Signal type expected.
        test_time: int default = 1E12
            Testing time in ps
        return_data: bool, default = False
            If True returns both the period and the data
        debug: bool, default = False
            If True, connects the channel to the internal timer

    Returns
    =======
        Period in ps, or a tuple containing the period in ps and the mesured data
    """
    if debug:
        tagger.setTestSignal(channel_number, True)
    tagger.setTriggerLevel(channel_number, signal_type.value.trigger_voltage)
    if signal_type.value.sense is SignalEdgeEnum.FALLEDGE:
        channel_number = -abs(channel_number)
    correlation = _TimeTagger.StartStop(
        tagger=tagger,
        click_channel=channel_number,
        binwidth=5,  # Poner 1
    )
    correlation.startFor(capture_duration=int(test_time))
    correlation.waitUntilFinished()
    data = _np.array(list(zip(*correlation.getData())))
    delay, jitter_rms = _get_delay_and_jitter(data[0], data[1])
    return (delay, data) if return_data else delay


def pulse_delay_calibration(
    tagger: _TimeTagger.TimeTagger,
    laser_channel: int,
    APD_channel: int,
    laser_channel_type: SignalTypeEnum = SignalTypeEnum.NIM,
    APD_channel_type: SignalTypeEnum = SignalTypeEnum.TTL,
    debug: int = None,
):
    """Calibrate pulse delay.

    Measures the delay between events in laser_channel and APD_channel.
    If debug is a non zero number, that delay (in ps) is set for APD_channel
    and both channels are connected to internal timer.

    Parameters
    ----------
    tagger: TimeTagger.TimeTagger
        Time tagger instance
    laser_channel : int
        Trigger (laser signal) channel number (positive).
    APD_channel : int
        signal (APD signal) channel number (positive).
    laser_channel_type, APD_channel_type: SignalTypeEnum
        Signal type expected on each channel.
    debug : int, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    if debug:
        tagger.setTestSignal(laser_channel, True)
        tagger.setTestSignal(APD_channel, True)
        tagger.setInputDelay(channel=APD_channel, delay=debug)
    tagger.setTriggerLevel(laser_channel, laser_channel_type.value.trigger_voltage)
    if laser_channel_type.value.sense is SignalEdgeEnum.FALLEDGE:
        laser_channel = -abs(laser_channel)
    tagger.setTriggerLevel(APD_channel, APD_channel_type.value.trigger_voltage)
    if APD_channel_type.value.sense is SignalEdgeEnum.FALLEDGE:
        APD_channel = -abs(APD_channel)
    correlation = _TimeTagger.StartStop(
        tagger=tagger,
        click_channel=APD_channel,
        start_channel=laser_channel,
        binwidth=2,  # poner 1 ps_
    )
    correlation.startFor(capture_duration=int(.5e12))
    correlation.waitUntilFinished()
    bins = correlation.getData()
    data = _np.array(list(zip(*bins)))
    return data


def get_tt_info() -> dict:
    """Get Time Tagger info.

    Returns dict by serial number.
    """
    serials = _TimeTagger.scanTimeTagger()
    # print(f'Found {len(serials)} connected Time Tagger(s)')
    # print("usando el primero")

    rv = {}
    for serial in serials:
        sub_dict = {}
        with _TimeTagger.createTimeTagger(serial) as tagger:
            sub_dict["model"] = tagger.getModel()
            sub_dict["lic_info"] = tagger.getDeviceLicense()
            sub_dict["channels"] = tagger.getChannelList(
                _TimeTagger.ChannelEdge.Rising
            )
            # if model not in ['Time Tagger Ultra', 'Time Tagger X']:
            #     raise ValueError(f'Currently {model} is not supported by this program')
        rv[serial] = sub_dict
    return rv


def get_channels_delay(
    tagger: _TimeTagger.TimeTagger, channels: int | _Sequence[int]
) -> list[int]:
    """Get hardware delays in ps for specified channels."""
    if isinstance(channels, int):
        channels = [
            channels,
        ]
    delays = [tagger.getDelayHardware(c) for c in channels]
    return delays


def set_channels_delay(
    tagger: _TimeTagger.TimeTagger,
    settings: tuple[int, int] | _Sequence[tuple[int, int]],
):
    """Set hardware delays in ps for specified channels."""
    if isinstance(settings[0], int):
        settings = [
            settings,
        ]
    [tagger.setDelayHardware(c, d) for c, d in settings]
    print([tagger.getDelayHardware(c) for c, _ in settings])


def set_channel_level(
        tagger: _TimeTagger.TimeTagger,
        channel: int,
        level_type: SignalTypeEnum,
        ):
    if level_type.value.sense is SignalEdgeEnum.FALLEDGE:
        channel = -abs(channel)
    tagger.setTriggerLevel(channel, level_type.value.trigger_voltage)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print(get_tt_info())
    with _TimeTagger.createTimeTagger() as tagger:
        laser_trigger_channel = 1
        APD_channel = 2
        data = pulse_delay_calibration(
            tagger, laser_trigger_channel, APD_channel, debug=600
        )
        x = data[0]
        h = data[1]
        plt.scatter(x, h)
        print(f"El delay parece ser de {x[_np.argmax(h)]} ps")

        mean, data = calibrate_period(
            tagger, laser_trigger_channel, return_data=True, debug=True
        )
        x = data[0]
        h = data[1]
        plt.scatter(x, h)
        print(f"El período parece ser de {x[_np.argmax(h)]} ps")
        print(_get_delay_and_jitter(x, h))
