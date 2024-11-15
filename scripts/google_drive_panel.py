import panel as pn
import pandas as pd
import hvplot.pandas
from bokeh.models import LinearAxis, Range1d
import holoviews as hv
import os
import numpy as np

# Enable Panel
hv.extension('bokeh')
pn.extension()

# File lists for EC, optics, and GIF data
ec_file_list = [
    '/Users/sarahpearce/Library/CloudStorage/GoogleDrive-sarah@tynt.io/Shared drives/Data/Devices/2024/10/20241014/20241014_LS_4564/cycle1/20241014_LS_4564_cycle1.tyntEC',
    '/Users/sarahpearce/Library/CloudStorage/GoogleDrive-sarah@tynt.io/Shared drives/Data/Devices/2024/10/20241014/20241014_LS_4564/cycle6/20241014_LS_4564_cycle6.tyntEC'
]

optics_file_list = [
    '/Users/sarahpearce/Library/CloudStorage/GoogleDrive-sarah@tynt.io/Shared drives/Data/Devices/2024/10/20241014/20241014_LS_4564/cycle1/20241014_LS_4564_cycle1.tyntOptics',
    '/Users/sarahpearce/Library/CloudStorage/GoogleDrive-sarah@tynt.io/Shared drives/Data/Devices/2024/10/20241014/20241014_LS_4564/cycle6/20241014_LS_4564_cycle6.tyntOptics'
]

gif_file_list = [
    '/Users/sarahpearce/Library/CloudStorage/GoogleDrive-sarah@tynt.io/Shared drives/Data/Devices/2024/10/20241014/20241014_LS_4565/cycle1/pictures/20241014_LS_4565_cycle1.gif'
]

# Helper function to create dropdown links with default option
def create_file_links(file_list):
    file_links = {"Select File": None}  # Add default "Select File" option
    for file_path in file_list:
        base_name = os.path.basename(file_path)
        name_without_ext = os.path.splitext(base_name)[0]
        file_links[name_without_ext] = file_path
    return file_links

ec_file_links = create_file_links(ec_file_list)
optics_file_links = create_file_links(optics_file_list)
gif_file_links = create_file_links(gif_file_list)

# Dropdown widgets
ec_dropdown = pn.widgets.Select(name='Select EC File', options=ec_file_links)
optics_dropdown = pn.widgets.Select(name='Select Optics File', options=optics_file_links)
gif_dropdown = pn.widgets.Select(name='Select GIF', options=gif_file_links)

# Placeholders for the interactive plots, GIF display, and source file paths
ec_plot_pane = pn.pane.HoloViews(object=hv.Curve([]))  # Empty Curve as placeholder
optics_plot_pane = pn.pane.HoloViews(object=hv.Curve([]))  # Empty Curve as placeholder
gif_pane = pn.pane.GIF(sizing_mode='stretch_both')
ec_source_pane = pn.pane.Markdown(object="No Data Selected")
optics_source_pane = pn.pane.Markdown(object="No Data Selected")
gif_source_pane = pn.pane.Markdown(object="No Data Selected")

# Callback function for EC data
def load_and_plot_ec(event):
    file_path = event.new
    if not file_path:
        ec_plot_pane.object = hv.Text("No Data Selected")
        ec_source_pane.object = "No Data Selected"
        return
    try:
        df = pd.read_csv(file_path, sep='\t', comment='#', index_col=0)
        time = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S.%f')
        start_time = time.iloc[0]
        time = (time - start_time).dt.total_seconds()
        df['Time'] = time

        current = df['Current (A)'] * 1000
        df['Current (mA)'] = current

        charge = df['Charge (C)']
        charge_plot = df.hvplot.line(x='Time', y='Charge (C)', color='green', label='Charge (C)')
        voltage_plot = df.hvplot.line(x='Time', y='Voltage (V)', color='blue', label='Voltage (V)')
        current_plot = df.hvplot.line(x='Time', y='Current (mA)', color='red', label='Current (mA)', yaxis='right')

        combined_plot = (charge_plot * voltage_plot * current_plot).opts(
            width=800,
            height=400,
            legend_position='right'
        )

        ec_plot_pane.object = combined_plot
        ec_source_pane.object = f"**Source:** {file_path}"
    except Exception as e:
        ec_plot_pane.object = hv.Text(f"Error: {e}")
        ec_source_pane.object = ""

# Callback function for Optics data
def load_and_plot_optics(event):
    file_path = event.new
    if not file_path:
        optics_plot_pane.object = hv.Text("No Data Selected")
        optics_source_pane.object = "No Data Selected"
        return
    try:
        df = pd.read_csv(file_path, sep='\t', comment='#', index_col=0)
        raw_time = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S.%f')
        shifted_time = (raw_time - raw_time[0]).dt.total_seconds()
        wavelengths = df.columns[1:-1].astype(float)  # Extract wavelengths as floats
        transmission_data = df.iloc[:, 1:-1].to_numpy()

        desired_wavelength = 550
        desired_index = np.abs(wavelengths - desired_wavelength).argmin()
        transmission = transmission_data[:, desired_index]

        optics_df = pd.DataFrame({'Time': shifted_time, 'Transmission (550 nm)': transmission})
        transmission_plot = optics_df.hvplot.line(
            x='Time', y='Transmission (550 nm)', color='grey',
            label=f'Transmission at {desired_wavelength} nm',
            ylabel='Transmission (%)'
        ).opts(width=800, height=400)

        optics_plot_pane.object = transmission_plot
        optics_source_pane.object = f"**Source:** {file_path}"
    except Exception as e:
        optics_plot_pane.object = hv.Text("")
        optics_source_pane.object = f"### Error\n```\n{e}\n```"

# Callback function for GIF display
def display_gif(event):
    file_path = event.new
    if not file_path:
        gif_pane.object = "No Data Selected"
        gif_source_pane.object = "No Data Selected"
        return
    try:
        gif_pane.object = file_path
        gif_source_pane.object = f"**Source:** {file_path}"
    except Exception as e:
        gif_pane.object = f"Error: {e}"
        gif_source_pane.object = ""

# Link callbacks to dropdowns
ec_dropdown.param.watch(load_and_plot_ec, 'value')
optics_dropdown.param.watch(load_and_plot_optics, 'value')
gif_dropdown.param.watch(display_gif, 'value')

# Dashboard layout
ec_section = pn.Column(
    pn.pane.Markdown("### Electrochemical Data Viewer"),
    ec_dropdown,
    ec_plot_pane,
    ec_source_pane  # Display the source file path
)

optics_section = pn.Column(
    pn.pane.Markdown("### Optics Data Viewer"),
    optics_dropdown,
    optics_plot_pane,
    optics_source_pane  # Display the source file path
)

gif_section = pn.Column(
    pn.pane.Markdown("### GIF Viewer"),
    gif_dropdown,
    gif_pane,
    gif_source_pane  # Display the source file path
)

data_viewer_content = pn.Row(ec_section, optics_section, gif_section)

# Use a Panel template
template = pn.template.FastListTemplate(
    title='Interactive Experiment Comparison Dashboard',
    main=data_viewer_content,
    accent_base_color="#00564a",
    header_background="#00564a",
)
template.show()
