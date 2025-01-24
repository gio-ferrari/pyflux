import TimeTagger as _TimeTagger
import numpy as _np
from typing import Sequence as _Sequence
from typing import Union, List, Tuple
from enum import Enum
from dataclasses import dataclass as _dataclass
import pathlib as _pathlib
from numba import njit

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
    tagger: _TimeTagger.TimeTagger, channels: Union[int, _Sequence[int]]
) -> List[int]:
    """Get hardware delays in ps for specified channels."""
    if isinstance(channels, int):
        channels = [
            channels,
        ]
    delays = [tagger.getDelayHardware(c) for c in channels]
    return delays


def set_channels_delay(
    tagger: _TimeTagger.TimeTagger,
    settings: Union[Tuple[int, int], _Sequence[Tuple[int, int]]],
):
    """Set hardware delays in ps for specified channels."""
    if isinstance(settings[0], int):
        settings = [
            settings,
        ]
    [tagger.setDelayHardware(c, d) for c, d in settings]
    print("SWABIAN HW time delays:", [tagger.getDelayHardware(c) for c, _ in settings])


def set_channel_level(
        tagger: _TimeTagger.TimeTagger,
        channel: int,
        level_type: SignalTypeEnum,
        ):
    if level_type.value.sense is SignalEdgeEnum.FALLEDGE:
        channel = -abs(channel)
    tagger.setTriggerLevel(channel, level_type.value.trigger_voltage)


@njit
def time_tags2delays(timestamps: _np.ndarray, channels: _np.ndarray,
                     overflow_types: _np.ndarray, APD_channel: int,
                     laser_channel: int, period: int,
                     last_timestamp: int,
                     ):
    """calculate ."""
    # last_timestamp = 0  # Puede ser que perdamos info, ponerlo en errors[1]
    n_errors = 0
    last_pos = 0
    rv = _np.empty((len(timestamps), ), dtype=_np.int64)
    for ts, chan, type_ in zip(timestamps, channels, overflow_types):
        # tag.type can be: 0 - TimeTag, 1- Error, 2 - OverflowBegin, 3 -
        # OverflowEnd, 4 - MissedEvents (you can use the TimeTagger.TagType IntEnum)
        if type_ != 0:
            # tag is not a TimeTag, so we are in an error state, e.g. overflow
            last_timestamp = 0
            n_errors += 1
        elif (chan == laser_channel) and (last_timestamp != 0):
            # valid event
            rv[last_pos] = period + last_timestamp
            rv[last_pos] -= ts
            if rv[last_pos] < 0:
                rv[last_pos] += period
            if rv[last_pos] < 0:
                print("Tiempo negativo", rv[last_pos]/period, ts, last_timestamp)
            last_pos += 1
        elif chan == APD_channel:
            last_timestamp = ts
        else:
            # print("unknown channel")
            n_errors += 1
    if n_errors:
        print("errores = ", n_errors)
    return rv[:last_pos], last_timestamp


def swabian2numpy(filename: str, period: int, APD_channel: int, laser_channel: int):
    """Extrae info de un archivo de swabian y lo graba en uno numpy."""
    # base_dir = _pathlib.Path.home() / "pMinflux_data"
    # base_dir.mkdir(parents=True, exist_ok=True)
    # date_str = _datetime.datetime.now().isoformat(
    #     timespec='seconds').replace('-', '').replace(':', '-')
    # out_file_name = base_dir / ('tcspc_data' + date_str + '.npy')
    original_filename = _pathlib.Path(filename)
    out_file_name = original_filename.with_suffix('.npy')
    filereader = _TimeTagger.FileReader(filename)
    n_events = int(5E6)
    i = 0
    batch = []
    lts = 0 
    while filereader.hasData():
        data = filereader.getData(n_events=n_events)
        channel = data.getChannels()            # The channel numbers
        timestamps = data.getTimestamps()       # The timestamps in ps
        overflow_types = data.getEventTypes()   # TimeTag = 0, Error = 1, OverflowBegin = 2, OverflowEnd = 3, MissedEvents = 4
        missed_events = data.getMissedEvents()  # The numbers of missed events in case of overflow
        # Output to table
        rv, lts = time_tags2delays(timestamps, channel, overflow_types,
                                      APD_channel, laser_channel, period, lts)
        batch.append(rv)
    with open(out_file_name, "wb") as fd:
        _np.save(fd, _np.concatenate(batch))


if __name__ == "__main__":
    input_file = r"C:\Users\Minflux\Documents\PythonScripts\reading_swabian\filename.ttbin"
    input_file = r"C:\Users\Minflux\Documents\Andi\pyflux\lefilename.ttbin"
    swabian2numpy(input_file,
                  50000, 4, 1)
    a = _np.load(r"C:\Users\Minflux\Documents\Andi\pyflux\lefilename.npy")
    import matplotlib.pyplot as plt
    plt.hist(a, 250, range=(-13,50000))
    plt.figure()
    plt.hist(a, 250, range=(a.min(), 0))

if False:
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
