#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 11:01:37 2024

@author: azelcer
"""

from configparser import ConfigParser as _ConfigParser
from dataclasses import dataclass as _dataclass, field as _field
from collections import OrderedDict as _OrderedDict
from typing import List


_CONFIG_FILENAME = "tcspc.ini"

@_dataclass
class APDInfo:
    channel: int
    delay: int
    signal_type: str  # Should be SignaltypeEnum


@_dataclass
class TCSPInstrumentInfo:
    """Information about a TCSPC intrument."""

    serial: str
    laser_channel: int = 0
    laser_signal: str = 'NIM'
    period: int = 0  # in ps
    shutter_delays: List[int] = _field(default_factory=list)
    APD_info: List[APDInfo] = _field(default_factory=list)

    def save(self, filename: str = _CONFIG_FILENAME):
        """Save data to file."""
        config = _ConfigParser()
        config["General"] = {
            "serial": self.serial,
            "laser_channel": self.laser_channel,
            "laser_signal": self.laser_signal,
            "period": self.period,
            "shutter_delays": ", ".join(str(_) for _ in self.shutter_delays),
        }
        for APDi in self.APD_info:
            config[f"APD.{APDi.channel}"] = {
                "delay": APDi.delay,
                "signal_type": APDi.signal_type,
                }
        with open(filename, "wt") as configfile:
            config.write(configfile)

    @classmethod
    def load(cls, filename: str = _CONFIG_FILENAME):
        """Load config.

        Raises not found
        """
        config = _ConfigParser()
        if not config.read(filename):
            raise FileNotFoundError
        sh_delays = config['General']['shutter_delays']
        sh_delays = [int(d) for d in sh_delays.split(',')] if sh_delays else []
        APD_info = [
            APDInfo(int(section_name.split('.')[1]),
                    config.getint(section_name, 'delay'),
                    config.get(section_name, 'signal_type'),
                    )
            for section_name in config.sections() if section_name.startswith('APD.')
            ]
        return cls(
            config['General']['serial'],
            config.getint('General', 'laser_channel'),
            config.get('General', 'laser_signal'),
            config.getint('General', 'period'),
            sh_delays,
            APD_info,
            )


if __name__ == '__main__':
    a = TCSPInstrumentInfo("dsfasdf", 1, 'TTL', 1266666, [123, 567, 234, 56],
                           [APDInfo(2, 300, 'NIM'), APDInfo(5, 12300, 'TTL')],)
    a.save("XXXXXXXXX.ini")
    b = TCSPInstrumentInfo.load("XXXXXXXXX.ini")
