#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt

from ..data import klarfio


def _coerce_columns_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert DataFrame columns to numeric where possible; non-numeric become NaN."""
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        except Exception:
            pass
    return df


class WaferMap:
    """
    Represents a wafer map containing defect information and relevant metadata.

    Args:
        defect_record (pd.DataFrame): 
            A DataFrame containing defect information.
            
        sample_center_location (tuple): 
            A reference coordinate where the 0,0 die has its origin.
            
        die_pitch (tuple): 
            The pitch between the die in the array.
            
        die_origin (tuple): 
            Coordinates of the lower left corner of the (0,0) die.
        
        sample_test_plan (array, optional):
            Dies to scan. If left blank, klarfkit will auto-calculate all dies from pitch, origin, and sample size
            
        orientation_marker (str, optional): 
            Location of the notch or flat of the wafer (e.g., 'NOTCH', 'FLAT'). Default is 'NOTCH'.
            
        orientation (str, optional): 
            Orientation of the wafer (e.g., 'UP', 'DOWN', 'LEFT', 'RIGHT'). Default is 'DOWN'.
            
        class_lookup (dict|None, optional): 
            A table of index numbers associated with defect descriptions. Default is None.
            
        file_version (str, optional): 
            Version of the file. Default is 1.1.
            
        file_timestamp (str, optional): 
            Timestamp of the file. Default is the current time.
            
        inspection_station (str, optional): 
            Inspection station information. Default is "klarfkit".
            
        sample_type (str, optional): 
            Type of the sample. Default is 'WAFER'.
            
        result_timestamp (str, optional): 
            Timestamp of the inspection result. Default is current time.
            
        lot_id (str, optional): 
            ID of the lot. Default is 'XXXX'.
            
        sample_size (int, optional): 
            Size of the sample in um. Default is 300,000.
            
        setup_id (str, optional): 
            ID of the setup. Default is None.
            
        step_id (str, optional): 
            ID of the step. Default is None.
            
        wafer_id (str, optional): 
            ID of the wafer. Default is 'XXXXXXXXXXX'.
            
        slot (str, optional): 
            Slot number of the wafer in the lot. Default is 1.

    Notes:
        - The CLASS LOOKUP table is used to associate index numbers with defect descriptions.
        - Each device can have multiple defects listed in the DEFECT RECORD section.
    """
    def __init__(self, defect_record: pd.DataFrame|None = None, sample_center_location = (150_000,150_000),
                 die_pitch = (10_000, 10_000), die_origin = (0,0), sample_test_plan = None, orientation_marker = 'NOTCH',
                 orientation = 'DOWN', class_lookup: dict|None = None, inspection_test: int = 1, area_per_test: float|None = None, 
                 file_version = '1 1', file_timestamp = None, inspection_station = '"klarfkit" "v0.0.4"', sample_type = 'WAFER',
                 result_timestamp = None, lot_id = 'XXXX', sample_size = 300_000, setup_id = None, step_id = None, wafer_id = 'XXXXXXXXXXX', slot = 1):
        self._die_pitch = die_pitch
        self._sample_size = sample_size
        self._die_origin = die_origin
        self._center_location = sample_center_location
        self._defect_list = defect_record

        if isinstance(defect_record, (pd.DataFrame)):
            self._calculate_actual_locations(defect_record)

        self.file_version = file_version
        if file_timestamp:
            self.file_timestamp = datetime.datetime.strptime(file_timestamp, "%m-%d-%y %H:%M:%S")
        else:
            self.file_timestamp = datetime.datetime.now()
        
        self.inspection_station = inspection_station
        self.sample_type = sample_type
        if result_timestamp:
            self.result_timestamp = datetime.datetime.strptime(result_timestamp, "%m-%d-%y %H:%M:%S")
        else:
            self.result_timestamp = datetime.datetime.now()

        self.lot_id = lot_id
        self.setup_id = setup_id
        self.step_id = step_id
        if orientation_marker.upper() in ['NOTCH','FLAT']:
            self.orientation_marker = orientation_marker.upper()
        else:
            raise ValueError(f"\nUnknown Orientation Marker: {orientation_marker.upper()}\nPlease use either NOTCH or FLAT")
        
        self.orientation = orientation
        self.class_lookup = class_lookup
        self.sample_test_plan = sample_test_plan
        self.wafer_id = wafer_id
        self.slot = slot

    @property
    def sample_size(self):
        return self._sample_size

    @sample_size.setter
    def sample_size(self, value):
        self._sample_size = value

    @property
    def die_pitch(self):
        return self._die_pitch

    @die_pitch.setter
    def die_pitch(self, value):
        if self._die_pitch:
            self._update_relative_coordinates(self._die_pitch, value, self._center_location, self._center_location)
        self._die_pitch = value

    @property
    def die_origin(self):
        return self._die_origin

    @die_origin.setter
    def die_origin(self, value):
        self._die_origin = value

    @property
    def center_location(self):
        return self._center_location

    @center_location.setter
    def center_location(self, value):
        if self._center_location:
            self._update_relative_coordinates(self._die_pitch, self._die_pitch, self._center_location, value)
        self._center_location = value

    @property
    def defect_list(self):
        return self._defect_list

    def _calculate_actual_locations(self, df):
        df['_XACTUAL'] = (df['XINDEX'] * self._die_pitch[0]) + df['XREL'] - self._center_location[0]
        df['_YACTUAL'] = (df['YINDEX'] * self._die_pitch[1]) + df['YREL'] - self._center_location[1]

    def _update_relative_coordinates(self, old_die_pitch, new_die_pitch, old_center, new_center):
        scale_x = new_die_pitch[0] / old_die_pitch[0]
        scale_y = new_die_pitch[1] / old_die_pitch[1]

        self._defect_list['XREL'] = (self._defect_list['_XACTUAL'] + new_center[0]) % new_die_pitch[0]
        self._defect_list['YREL'] = (self._defect_list['_YACTUAL'] + new_center[1]) % new_die_pitch[1]

        self._defect_list['XINDEX'] = ((self._defect_list['_XACTUAL'] + new_center[0]) // new_die_pitch[0]).astype(int)
        self._defect_list['YINDEX'] = ((self._defect_list['_YACTUAL'] + new_center[1]) // new_die_pitch[1]).astype(int)
    
    def to_klarf(
        self, 
        file_path, 
        version = '1.1'):
        
        raise NotImplemented("Coming soon")
        
        
    def __repr__(self):
        fig, ax = plt.subplots(figsize=(3,3))
        
        circle = plt.Circle(
            (0,0),
            radius=self.sample_size/2, 
            alpha = 1 , 
            color = 'gray', 
            fill = False)
        
        ax.add_artist(circle)
        
        for i in range(-int(self.sample_size // self.die_pitch[0]) - 10,
                       int(self.sample_size // self.die_pitch[0]) + 10):

            die_x_lines = -self._center_location[0] + (i * self._die_pitch[0])
            height_squared = (self.sample_size / 2) ** 2 - die_x_lines ** 2
            if height_squared > 0:
                height = np.sqrt(height_squared)

                ax.vlines(die_x_lines,
                           -height,
                           height,
                           color='gray',
                           alpha=0.2)

        for i in range(-int(self.sample_size // self.die_pitch[1]) - 10,
                       int(self.sample_size // self.die_pitch[1]) + 10):
            die_y_lines = -self._center_location[1] + (i * self._die_pitch[1])
            width_squared = (self.sample_size / 2) ** 2 - die_y_lines ** 2
            if width_squared > 0:
                width = np.sqrt(width_squared)

                ax.hlines(die_y_lines,
                           -width,
                           width,
                           color='gray',
                           alpha=0.2)

        if isinstance(self._defect_list, (pd.DataFrame)):
            ax.scatter(
                x=self._defect_list['_XACTUAL'],
                y=self._defect_list['_YACTUAL'],
                color='red', s = 4)

        ax.set_xlim(-self.sample_size / 2 - 2000, self.sample_size / 2 + 2000)
        ax.set_ylim(-self.sample_size / 2 - 2000, self.sample_size / 2 + 2000)
        ax.spines[['right', 'top','left','bottom']].set_visible(False)
        ax.set_yticks([])
        ax.set_xticks([])
        
        return ""
        
    @classmethod
    def read_klarf(cls, file_path) -> 'WaferMap':
        """Read a KLARF file (v1.2 or v1.8) using klarfio as the parser backend."""
        parsed = klarfio.klarf(filename=str(file_path))

        if parsed.version == "1.8":
            return cls._from_klarfio_v1_8(parsed)
        elif parsed.version == "1.2":
            return cls._from_klarfio_v1_2(parsed)
        else:
            raise ValueError(f"Unsupported KLARF version: {parsed.version}")

    @classmethod
    def _from_klarfio_v1_8(cls, parsed) -> 'WaferMap':
        file_data = parsed.data["FileRecord_1.8"]

        # Find first LotRecord
        lot_keys = [k for k in file_data if k.startswith("LotRecord")]
        if not lot_keys:
            raise ValueError("No LotRecord found in v1.8 KLARF")
        lot_data = file_data[lot_keys[0]]

        # Find first WaferRecord
        wafer_keys = [k for k in lot_data if k.startswith("WaferRecord")]
        if not wafer_keys:
            raise ValueError("No WaferRecord found in v1.8 KLARF")
        wafer_data = lot_data[wafer_keys[0]]

        # Defect list: Columns + Data → DataFrame
        defect_list = wafer_data["DefectList"]
        columns = [c["Column"] for c in defect_list["Columns"]]
        defect_record = pd.DataFrame(defect_list["Data"], columns=columns)
        defect_record = _coerce_columns_to_numeric(defect_record)

        # Metadata
        die_pitch = tuple(float(v) for v in lot_data.get("DiePitch", [10000, 10000]))
        die_origin = tuple(float(v) for v in wafer_data.get("DieOrigin", [0, 0]))
        sample_size = float(lot_data.get("SampleSize", [300000])[0])
        center_location = tuple(float(v) for v in wafer_data.get("SampleCenterLocation", [sample_size / 2, sample_size / 2]))

        return cls(
            defect_record=defect_record,
            sample_center_location=center_location,
            die_pitch=die_pitch,
            die_origin=die_origin,
            sample_size=sample_size,
            file_version="1 8",
        )

    @classmethod
    def _from_klarfio_v1_2(cls, parsed) -> 'WaferMap':
        data = parsed.data["FileRecord_1.2"]

        # Defect list
        defect_list = data["DefectList"]
        columns = [c["Column"] for c in defect_list["Columns"]]
        defect_record = pd.DataFrame(defect_list["Data"], columns=columns)
        defect_record = _coerce_columns_to_numeric(defect_record)

        # Metadata — klarf v1.2 stores SampleSize in mm, convert to um
        die_pitch = tuple(float(v) for v in data.get("DiePitch", [10000, 10000]))
        sample_size_raw = float(data.get("SampleSize", [300])[0])
        sample_size = sample_size_raw * 1000 if sample_size_raw < 10000 else sample_size_raw
        center_location = tuple(float(v) for v in data.get("SampleCenterLocation", [sample_size / 2, sample_size / 2]))

        return cls(
            defect_record=defect_record,
            sample_center_location=center_location,
            die_pitch=die_pitch,
            sample_size=sample_size,
            file_version="1 2",
        )
    
    
    def describe(self):
        attrs = {
            'file_version': 'File Version',
            'file_timestamp': 'File Timestamp',
            'inspection_station': 'Inspection Station',
            'sample_type': 'Sample Type',
            'result_timestamp': 'Result Timestamp',
            'lot_id': 'Lot ID',
            'slot': 'Slot',
            'wafer_id': 'Wafer ID',
            'sample_size': 'Sample Size',
            'setup_id': 'Setup ID',
            'step_id': 'Step ID',
            'orientation': 'Orientation',
            'orientation_marker': 'Orientation Marker',
            'die_pitch': 'Die Pitch',
            'die_origin': 'Die Origin',
            '_center_location': 'Center Location',
        #    'sample_test_plan': 'Sample Test Plan',
        #    'defect_count': 'Defect Count',
        }

        for attr, desc in attrs.items():
            value = getattr(self, attr)
            print(f"{desc: <22} | {value}")

        #self.print_attr_desc('sample_test_plan', f"{len(self.sample_test_plan)} sites")
        print(f"{'Defect Count':<22} | {len(self._defect_list)}")


        
    def plot_wafer_map(self, color='red', die_line_alpha=0.2, die_line_color='gray', *args, **kwargs):
        theta = np.linspace(0, 2*np.pi, 100)
        radius = self.sample_size / 2
        wafer_x, wafer_y = radius * np.cos(theta), radius * np.sin(theta)
        
        range_x = range(-int(self.sample_size // self.die_pitch[0]) - 10, int(self.sample_size // self.die_pitch[0]) + 10)
        range_y = range(-int(self.sample_size // self.die_pitch[1]) - 10, int(self.sample_size // self.die_pitch[1]) + 10)

        for i in range_x:
            die_x_lines = -self.center_location[0] + (i * self.die_pitch[0])
            height_squared = radius**2 - die_x_lines**2
            if height_squared > 0:
                height = np.sqrt(height_squared)
                plt.vlines(die_x_lines, -height, height, color=die_line_color, alpha=die_line_alpha)

        for i in range_y:
            die_y_lines = -self.center_location[1] + (i * self.die_pitch[1])
            width_squared = radius**2 - die_y_lines**2
            if width_squared > 0:
                width = np.sqrt(width_squared)
                plt.hlines(die_y_lines, -width, width, color=die_line_color, alpha=die_line_alpha)

        if color in self.defect_list.columns:
            scatter_colors = self.defect_list[color]
            plt.scatter(self.defect_list['_XACTUAL'], self.defect_list['_YACTUAL'], c=scatter_colors, *args, **kwargs)
            plt.legend(loc='lower left')
        else:
            plt.scatter(self.defect_list['_XACTUAL'], self.defect_list['_YACTUAL'], color=color, *args, **kwargs, zorder = 10)

        plt.plot(wafer_x, wafer_y, color='k', linewidth=0.3)
        plt.axis('equal')
