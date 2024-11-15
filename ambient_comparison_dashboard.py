import panel as pn
import pandas as pd
import psycopg2
import hvplot.pandas
import holoviews as hv
from holoviews import opts
import os
import glob
import numpy as np
import sys
import re
import holoviews as hv
import pandas as pd
from holoviews import opts
import numpy as np
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QComboBox, QPushButton, QLabel
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QComboBox, QPushButton, QLabel, QCheckBox, QScrollArea, QFormLayout
from psycopg2 import sql
import sys
from pathlib import Path
import re
import panel as pn
from PIL import Image
import io
from scripts.tynt_panel_trd_functions import *
from scripts.tynt_panel_baseline_functions import *
from scripts.comparison_functions import *
from scripts.html_functions import *
import panel as pn
from PIL import Image
import os

# Sarah's mac:  source dashboard_env/bin/activate  


hv.extension('bokeh')
pn.extension()

def connect_to_local(dbname='postgres', user='postgres', password='postgres', host='database.tynt.io', port='8001'):
    """
    Establish a connection to the local database and return the connection and cursor.
    
    Parameters:
    dbname (str): Name of the database.
    user (str): Username for the database.
    password (str): Password for the database.
    host (str): Hostname of the database server.
    port (str): Port number of the database server.
    
    Returns:
    conn: Database connection object.
    cursor: Database cursor object.
    """
    try:
        print('Establishing connection...')
        conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port,
                                gssencmode='disable', sslmode='disable', connect_timeout=2)
        print('Connection established')
        cursor = conn.cursor()
        return conn, cursor
    except OperationalError as e:
        print(f"The error '{e}' occurred")
        return None, None

def get_all_tables(conn, cursor):
    if conn and cursor:
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            """)
            # Fetch all the results
        tables = cursor.fetchall()
        # print("Tables in the database:")
        # for table in tables:
            # print(table[0])
        # conn.close() # don't close here, as other functions need it. use finally
    return tables

def get_eccheckins(conn, cursor):
    if conn and cursor:
        sql_query = '''
            SELECT * 
            FROM tyntdatabase_eccheckin
            LIMIT ALL 
            OFFSET 0;
        '''
        cursor.execute(sql_query)
        eccheckins = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        eccheckins = pd.DataFrame(eccheckins, columns=column_names)

    return eccheckins

def get_trd_eccheckins(conn, cursor, device_id_list):
    if conn and cursor:
        device_id_list = [int(id) for id in device_id_list]
        print(device_id_list)
        # Convert the list of IDs to a format suitable for SQL IN clause
        format_strings = ','.join(['%s'] * len(device_id_list))
        print(format_strings)
        sql_query = f'''
            SELECT * 
            FROM tyntdatabase_eccheckin
            WHERE device_id IN ({format_strings})
            LIMIT ALL 
            OFFSET 0;
        '''
        # Execute the query with the list of IDs as parameters
        cursor.execute(sql_query, device_id_list)
        trd_eccheckins = cursor.fetchall()
        print(trd_eccheckins)
        column_names = [desc[0] for desc in cursor.description]
        trd_eccheckins_df = pd.DataFrame(trd_eccheckins, columns=column_names)

    return trd_eccheckins_df

def get_devices(conn, cursor):
    if conn and cursor:
        sql_query = '''
            SELECT * 
            FROM tyntdatabase_device
            LIMIT ALL 
            OFFSET 0;
        '''
        cursor.execute(sql_query)
        devices = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        devices = pd.DataFrame(devices, columns=column_names)

    return devices

def get_trds(conn, cursor):
    if conn and cursor:
        sql_query = '''
            SELECT * 
            FROM tyntdatabase_trd
            LIMIT ALL 
            OFFSET 0;
        '''
        cursor.execute(sql_query)
        trds = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        trds = pd.DataFrame(trds, columns=column_names)

    return trds

def search_trd_name(df, search_string):
    # Ensure 'trd_name' is treated as string
    df['trd_name'] = df['trd_name'].astype(str)
    
    # Filter rows where 'trd_name' exactly matches the search string
    filtered_df = df[df['trd_name'] == search_string]
    
    # Extract corresponding 'id' values
    ids = filtered_df['id'].tolist()
    
    return ids

def get_deviceid_devices(conn, cursor, device_ids):
    if conn and cursor and device_ids:
        # Create a placeholder string for the number of device_ids
        placeholders = ', '.join(['%s'] * len(device_ids))
        
        # Update SQL query to filter by multiple device_ids
        sql_query = sql.SQL('''
            SELECT * 
            FROM tyntdatabase_device
            WHERE id IN ({})
            LIMIT ALL 
            OFFSET 0;
        ''').format(sql.SQL(placeholders))
        
        # Execute the query with device_ids as the parameters
        cursor.execute(sql_query, device_ids)
        devices = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        devices = pd.DataFrame(devices, columns=column_names)

    return devices

def get_trd_devices(conn, cursor, trd_id):
    if conn and cursor:
        # Update SQL query to filter by trd_id
        sql_query = '''
            SELECT * 
            FROM tyntdatabase_device
            WHERE trd_id = %s
            LIMIT ALL 
            OFFSET 0;
        '''
        # Execute the query with trd_id parameter
        cursor.execute(sql_query, (trd_id,))
        devices = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        devices = pd.DataFrame(devices, columns=column_names)

    return devices


def get_device_with_related_electrolyte(conn, cursor, device_id):
    # Query to get the device
    cursor.execute("SELECT * FROM device_device WHERE id = %s", [device_id])
    device = cursor.fetchone()

    # Query to get related WorkingElectrodeGlassBatch objects
    cursor.execute('''
        SELECT weg.*
        FROM device_electrolytebatch AS dweg
        INNER JOIN electrolytebatch AS weg ON dweg.electrolytebatch_id = weg.id
        WHERE dweg.device_id = %s;
    ''', [device_id])
    electrolytes = cursor.fetchall()

    return device, electrolytes


def find_directory(base_path, target_directory):
    """
    Search for the target directory starting from the base path.

    Args:
        base_path (str): The base path to start searching.
        target_directory (str): The name of the target directory to find.

    Returns:
        str: The path to the target directory if found, otherwise None.
    """
    for root, dirs, files in os.walk(base_path):
        for dir_name in dirs:
            if dir_name == target_directory:
                return os.path.join(root, dir_name)
    return None


from pathlib import Path

def extract_after_substring(path, substring):
    """
    Extract everything after the specified substring in the given path.

    Args:
        path (str): The full path to search within.
        substring (str): The substring to find in the path.

    Returns:
        str: The portion of the path after the substring, or None if substring is not found.
    """
    #path_obj = Path(path)
    # Convert the path to a string and find the position of the substring
    #path_str = str(path_obj)

    path_str = path # should work on both windows and mac
    
    index = path_str.find(substring)
    if index == -1:
        return None  # Substring not found
    
    # Extract everything after the substring and handle leading slashes
    result = path_str[index + len(substring):].lstrip('/')
    
    return result


def get_local_paths(path_list):
    local_paths = []
    print('accessing local paths', path_list)
    for path in path_list:
        print('path to split', path)
        substring = 'google-drive-data/'
        print('substring', substring)
        end_of_path = extract_after_substring(path, substring)
        print("Extracted Path:", end_of_path)

        # Find the 'Shared drives/Data' directory
        directory_path = search_shared_drives_data()
        print("Path to 'Shared drives/Data':", directory_path)

        final_path = combine_paths(directory_path, end_of_path)

        print("Final Path:", final_path)

        final_path_folder = str(Path(final_path).parent)
        print(final_path_folder)
        # Check if the final path exists
        if check_path_exists(final_path_folder):
            print(f"The path exists: {final_path_folder}")
        else:
            print(f"The path does not exist: {final_path_folder}")
        local_paths.append(final_path)

    return local_paths


def get_initial_photo_path(local_paths):
    initial_photo_paths = []
    for path in local_paths:
        directory_path = os.path.dirname(path)
        # Split the path into components
        path_parts = directory_path.split(os.path.sep)
        # Remove the last folder from the path
        path_parts.pop()
        # Append the new folder name
        path_parts.append("precycle0")
        path_parts.append("pictures")
        # Join the path components back together
        new_directory_path = os.path.sep.join(path_parts)

        initial_photo_paths.append(new_directory_path)
        return initial_photo_paths

from PIL import Image, ImageDraw, ImageFont

def create_blank_image_with_text(text, image_path):
    # Define image size and color
    width, height = 800, 600
    background_color = (255, 255, 255)  # White background
    text_color = (0, 0, 0)  # Black text

    # Create a new blank image
    image = Image.new('RGB', (width, height), background_color)
    draw = ImageDraw.Draw(image)


    # Calculate text size and position
    bbox = draw.textbbox((0, 0), text, font=ImageFont.load_default())
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    position = ((width - text_width) / 2, (height - text_height) / 2)

    # Draw text on the image
    draw.text(position, text, fill=text_color, font=ImageFont.load_default())

    # Save the image
    image.save(image_path)

    return image_path 

import os

def get_all_file_paths(folder_path):
    """
    Get the file paths of all files in the specified folder.

    Parameters:
    - folder_path: str, path to the folder containing the files
    - file_extension: str or None, file extension to filter by (default is None, meaning all files)

    Returns:
    - list: A list of file paths
    """
    file_paths = []
    # List all files in the directory
    try:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            # Check if it is a file
            if os.path.isfile(file_path):
                # Check file extension if specified
                    file_paths.append(file_path)
    except Exception as e:
        print(f"Error accessing folder {folder_path}: {e}")

    return file_paths

    
" ####################################### PANEL FUNCTIONS ######################################### "

def create_plot(data, x, y, color_col, title):
    """
    Create an interactive line plot with color based on a column.

    Parameters:
    - data: pandas DataFrame with the data to plot
    - x: column name for x-axis
    - y: column name for y-axis
    - color_col: column name for coloring the lines
    - title: title of the plot
    """
        # Convert color_col to string to avoid issues with non-string labels
    data[color_col] = data[color_col].astype(str)
    # Group data by the color column and create a line plot for each group
    # Group data by the color column and create a line plot for each group
    grouped = data.groupby(color_col)
    plots = []
    for name, group in grouped:
        # Use the name as label, converting to string if necessary
        plots.append(group.hvplot.line(x=x, y=y, label=str(name)))
    
    # Overlay the plots
    plot = hv.Overlay(plots).opts(
        opts.Curve(width=800, height=400, tools=['hover']),
        opts.Overlay(legend_position='right')
    )

    # Create a Panel dashboard with the plot
    dashboard = pn.Column(
        title,
        plot
    )

    # Return the Panel object
    return dashboard

" ############### FANCY VERSION OF PANEL FUNCITON #################### "
import pandas as pd
import hvplot.pandas  # Ensure hvplot extension is loaded
import holoviews as hv
from holoviews import opts
import panel as pn
import holoviews as hv
import hvplot.pandas
from holoviews import opts

pn.extension()

def make_single_plot(data, x, y, color_col1, title):
    data[color_col1] = data[color_col1].astype(str)
    # Create plots for the first color column
    grouped1 = data.groupby(color_col1)
    plots1 = []
    scatter_plots1 = []
    line_plots1 = []

    for name, group in grouped1:
        # Create a scatter plot for each group
        scatter_plot = group.hvplot.scatter(size=1, x=x, y=y, label=str(name))
        scatter_plots1.append(scatter_plot)


def make_row_traces(inputs):
    # Unpack
    data, x, y, color_col1, color_col2, title = inputs
    # Convert color columns to string type for grouping
    data[color_col1] = data[color_col1].astype(str)
    data[color_col2] = data[color_col2].astype(str)

    # Create plots for the first color column
    grouped1 = data.groupby(color_col1)
    plots1 = []
    scatter_plots1 = []
    line_plots1 = []

    for name, group in grouped1:
        # Create a scatter plot for each group
        scatter_plot = group.hvplot.scatter(size=1, x=x, y=y, label=str(name))
        scatter_plots1.append(scatter_plot)

    # Combine scatter and line plots into an overlay
    plot1 = hv.Overlay(scatter_plots1 + line_plots1).opts(
    opts.Scatter(size=3, tools=['hover']),
    opts.Overlay(legend_position='right', title=f'{title} - {color_col1}'))

    plot1_bokeh = hv.render(plot1, backend='bokeh')
    plot1_bokeh.legend.title = color_col1
    
    grouped2 = data.groupby(color_col2)
    plots2 = []
    for name, group in grouped2:
        plots2.append(group.hvplot.line(x=x, y=y, label=str(name)))
    
    plot2 = hv.Overlay(plots2).opts(
        opts.Curve(width=800, height=400, tools=['hover']),
        opts.Overlay(legend_position='right', title=f'{title} - {color_col2}')
        )
    
            # Convert holoviews to bokeh and manually adjust legends
    # Convert holoviews to bokeh
    plot2_bokeh = hv.render(plot2, backend='bokeh')
    plot2_bokeh.legend.title = color_col2

    return plot1, plot1_bokeh, plot2, plot2_bokeh

def make_row_bar(inputs):
    # Unpack
    data, x, y, color_col1, color_col2, title = inputs
    # Convert color columns to string type for grouping
    data[color_col1] = data[color_col1].astype(str)
    data[color_col2] = data[color_col2].astype(str)

    # Create bar plots with dots for the first color column
    grouped1 = data.groupby(color_col1)
    bar_plots1 = []
    for name, group in grouped1:
        bar_plot = group.hvplot.bar(x=x, y=y, label=str(name))
        dot_plot = group.hvplot.scatter(x=x, y=y, label=f'{name} dots', color='black')
        bar_plots1.append(bar_plot * dot_plot)
    
    plot1 = hv.Overlay(bar_plots1).opts(
        opts.Bars(width=800, height=400, tools=['hover']),
        opts.Scatter(size=10, color='black'),
        opts.Overlay(legend_position='right', title=f'{title} - {color_col1}')
    )
    
    # Convert holoviews to bokeh
    plot1_bokeh = hv.render(plot1, backend='bokeh')
    plot1_bokeh.legend.title = color_col1
    
    # Create bar plots with dots for the second color column
    grouped2 = data.groupby(color_col2)
    bar_plots2 = []
    for name, group in grouped2:
        bar_plot = group.hvplot.bar(x=x, y=y, label=str(name))
        dot_plot = group.hvplot.scatter(x=x, y=y, label=f'{name} dots', color='black')
        bar_plots2.append(bar_plot * dot_plot)
    
    plot2 = hv.Overlay(bar_plots2).opts(
        opts.Bars(width=800, height=400, tools=['hover']),
        opts.Scatter(size=10, color='black'),
        opts.Overlay(legend_position='right', title=f'{title} - {color_col2}')
    )
    
    # Convert holoviews to bokeh and manually adjust legends
    plot2_bokeh = hv.render(plot2, backend='bokeh')
    plot2_bokeh.legend.title = color_col2

    return plot1, plot1_bokeh, plot2, plot2_bokeh



def make_row_box(inputs):
    # Unpack
    data, x, y, color_col1, color_col2, title = inputs
    
    # Convert color columns to string type for grouping
    data[color_col1] = data[color_col1].astype(str)
    data[color_col2] = data[color_col2].astype(str)
    
    # Create box plots with dots for the first color column
    grouped1 = data.groupby(color_col1)
    box_plots1 = []
    scatter_plots1 = []
    
    for name, group in grouped1:
        box_plot = hv.BoxWhisker(group, kdims=x, vdims=y).opts(
            opts.BoxWhisker(width=400, height=200, tools=['hover'], box_color='blue')
        )
        dot_plot = hv.Scatter(group, kdims=x, vdims=y).opts(
            opts.Scatter(size=5, color='black')
        )
        box_plots1.append(box_plot)
        scatter_plots1.append(dot_plot)
    
    plot1 = hv.Overlay(box_plots1 + scatter_plots1).opts(
        opts.Overlay(legend_position='right', title=f'{title} - {color_col1}')
    )
    
    # Create box plots with dots for the second color column
    grouped2 = data.groupby(color_col2)
    box_plots2 = []
    scatter_plots2 = []
    
    for name, group in grouped2:
        box_plot = hv.BoxWhisker(group, kdims=x, vdims=y).opts(
            opts.BoxWhisker(width=400, height=200, tools=['hover'], box_color='blue')
        )
        dot_plot = hv.Scatter(group, kdims=x, vdims=y).opts(
            opts.Scatter(size=5, color='black')
        )
        box_plots2.append(box_plot)
        scatter_plots2.append(dot_plot)
    
    plot2 = hv.Overlay(box_plots2 + scatter_plots2).opts(
        opts.Overlay(legend_position='right', title=f'{title} - {color_col2}')
    )
    
    return plot1, plot2

def crop_image(image_path, crop_box):
    with Image.open(image_path) as img:
        cropped_img = img.crop(crop_box)
        buffer = io.BytesIO()
        cropped_img.save(buffer, format='PNG')
        return buffer.getvalue()




" ####################################### CONNECT AND GET DATA ######################################### "

def main():
    conn, cursor = connect_to_local()
    all_baselines_df = get_baselines(conn, cursor)
    print(all_baselines_df)
    all_baselines_list = reversed(all_baselines_df['baseline_version'].values)
    print(all_baselines_df.columns)
    all_trds_df = get_trds(conn, cursor)


    # selected_baseline_name, selected_devices = create_dynamic_baseline_dialog(all_baselines_df, conn, cursor)
    selected_comparison_type, selected_name_1, selected_name_2, selected_devices_1, selected_devices_2 = comparison_create_dynamic_comparison_dialog(all_baselines_df, all_trds_df, conn, cursor)
    print(f"Selected Comparison: {selected_comparison_type}")
    print(f"Selected Devices (1): {selected_devices_1}")
    print(f"Selected Devices (2): {selected_devices_2}")

    comparison_mapping = {
    'Baseline vs TRD': ('Baseline', 'TRD'),
    'TRD vs TRD': ('TRD', 'TRD'),
    'Baseline vs Baseline': ('Baseline', 'Baseline')}
    result = comparison_mapping.get(selected_comparison_type)
    if result:
        first_option = result[0]  # Access the first string
        second_option = result[1]  # Access the second string
        print(f"The first option is: {first_option}")
        print(f"The second option is: {second_option}")

    if selected_name_1 and selected_devices_1 and selected_name_2 and selected_devices_2:
        print(f"Search String Entered: {selected_name_1}")
        if first_option == 'Baseline':
            sub_df = all_baselines_df[all_baselines_df['baseline_version'] == selected_name_1]
            notes_string_1 = sub_df['notes'].values[0]
            # Get ID and Devices
            matching_ids = search_baseline_name(all_baselines_df, selected_name_1)
            baseline_id = matching_ids[0]
            baseline_devices = get_baseline_devices(conn, cursor, baseline_id)
            devices_df_1 = baseline_devices 
            # Get list of IDs for the specified device names
            device_id_list_1 = baseline_devices.loc[baseline_devices['device_name'].isin(selected_devices_1), 'id'].tolist()
            print('IDs corresponding to selected devices 1: ', device_id_list_1)
            device_list_1 = selected_devices_1
            print('device list 1: ', device_id_list_1)

        if first_option == 'TRD':
            sub_df = all_trds_df[all_trds_df['trd_name'] == selected_name_1]
            notes_string_1 = sub_df['notes'].values[0]
            # Get ID and Devices
            matching_ids = search_trd_name(all_trds_df, selected_name_1)
            trd_id = matching_ids[0]
            trd_devices = get_trd_devices(conn, cursor, trd_id)
            devices_df_1 = trd_devices 
            # Get list of IDs for the specified device names
            device_id_list_1 = trd_devices.loc[trd_devices['device_name'].isin(selected_devices_1), 'id'].tolist()
            print('IDs corresponding to selected devices 1: ', device_id_list_1)
            device_list_1 = selected_devices_1
            print('device list 1: ', device_id_list_1)

        if second_option == 'Baseline':
            sub_df = all_baselines_df[all_baselines_df['baseline_version'] == selected_name_2]
            notes_string_2 = sub_df['notes'].values[0]
            # Get ID and Devices
            matching_ids = search_baseline_name(all_baselines_df, selected_name_2)
            baseline_id = matching_ids[0]
            baseline_devices = get_baseline_devices(conn, cursor, baseline_id)
            devices_df_2 = baseline_devices 
            # Get list of IDs for the specified device names
            device_id_list_2 = baseline_devices.loc[baseline_devices['device_name'].isin(selected_devices_2), 'id'].tolist()
            print('IDs corresponding to selected devices 2: ', device_id_list_2)
            device_list_2 = selected_devices_2
            print('device list 2: ', device_id_list_2)

        if second_option == 'TRD':
            sub_df = all_trds_df[all_trds_df['trd_name'] == selected_name_2]
            notes_string_2 = sub_df['notes'].values[0]
            # Get ID and Devices
            matching_ids = search_trd_name(all_trds_df, selected_name_2)
            trd_id = matching_ids[0]
            trd_devices = get_trd_devices(conn, cursor, trd_id)
            devices_df_2 = trd_devices 
            # Get list of IDs for the specified device names
            device_id_list_2 = trd_devices.loc[trd_devices['device_name'].isin(selected_devices_2), 'id'].tolist()
            print('IDs corresponding to selected devices 2: ', device_id_list_2)
            device_list_2 = selected_devices_2
            print('device list 2: ', device_id_list_2)


        current_directory = os.getcwd()
        # cv_image_path = os.path.join(current_directory, '/figures/no_initial_photo_available.jpg')
        # no_data_image_path = os.path.join(current_directory, '/figures/no_data.png')


        ' ################################ BUILD ENTIRE DASHBOARD AS EMPTY FIRST #################################### '

        ' ###### SIDEBAR ###### '

        '#### PAGE 1: EMPTIES ####### '
        keyence_paths = ['In Progress']
        local_warmup_paths = ['In Progress']
        optics_folder_paths = ['In Progress']
        photo_folder_paths = ['In Progress']
        arbin_paths = ['None Found']

        '#### PAGE 2: EMPTIES ####### '
        all_folders_list = []
        photo_folder_paths = []

        '#### PAGE 3: EMPTIES ####### '
        '#### PAGE 4: EMPTIES ####### '
        '#### PAGE 5: EMPTIES ####### '

        '#### PAGE 1: GET DATA ####### '

        ' ############################# CALLING ALL HTML FUNCTIONS FOR DEVICE OVERVIEW TAB ################### '
        # adding route_name to baseline devices dataframe
        routes_df = get_routes(conn, cursor)

        # TABLE ONE
        data = devices_df_1 
        devices_df_1['route_id'] = [0 if x is None else x for x in devices_df_1['route_id'].values]
        new_row = pd.DataFrame({'route_id': [0], 'route_name': ['Unspecified']})
        routes_df = pd.concat([new_row, routes_df], ignore_index=True)
        devices_df_1  = pd.merge(devices_df_1, routes_df, on='route_id')
        print('FINAL TABLE OF DEVICE DATA', devices_df_1.columns)
        table_html_1 = all_devices_table(devices_df_1)
        selected_devices_1 = devices_df_1 

        # TABLE TWO
        data = devices_df_2
        devices_df_2['route_id'] = [0 if x is None else x for x in devices_df_2['route_id'].values]
        new_row = pd.DataFrame({'route_id': [0], 'route_name': ['Unspecified']})
        routes_df = pd.concat([new_row, routes_df], ignore_index=True)
        devices_df_2  = pd.merge(devices_df_2, routes_df, on='route_id')
        print('FINAL TABLE OF DEVICE DATA', devices_df_2.columns)
        table_html_2 = all_devices_table(devices_df_2)
        selected_devices_2 = devices_df_2

        # Bullet list of gathered data
        warmups = local_warmup_paths
        o_checks = optics_folder_paths
        p_checks = photo_folder_paths
        arbin = arbin_paths
        keyence = keyence_paths
        report_html = generate_devices_report(warmups, o_checks, p_checks, arbin, keyence)

        '#### PAGE 2: GET DATA ####### '

        # Getting path list so we can find photo data
        eccheckins_1 = get_trd_eccheckins(conn, cursor, device_id_list_1)
        path_list_1 = eccheckins_1['server_path'].values 
        path_list_1 = [item for item in path_list_1 if item is not None]
        ec_local_all_paths = get_local_paths(path_list_1)
        print('Device set 1 ec path list:', ec_local_all_paths)

        opticscheckins_1 = get_trd_opticscheckins(conn, cursor, device_id_list_1)
        path_list_1 = opticscheckins_1['server_path'].values 
        path_list_1 = [item for item in path_list_1 if item is not None]
        optics_local_all_paths = get_local_paths(path_list_1)
        print('Device set 1 optics path list:', optics_local_all_paths)

        # Path lists for set 2
        eccheckins_2 = get_trd_eccheckins(conn, cursor, device_id_list_2)
        path_list_2 = eccheckins_2['server_path'].values 
        path_list_2 = [item for item in path_list_2 if item is not None]
        ec_local_all_paths_2 = get_local_paths(path_list_2)
        print('Device set 2 ec path list:', ec_local_all_paths_2)

        opticscheckins_2 = get_trd_opticscheckins(conn, cursor, device_id_list_2)
        path_list_2 = opticscheckins_2['server_path'].values 
        path_list_2 = [item for item in path_list_2 if item is not None]
        optics_local_all_paths_2 = get_local_paths(path_list_2)
        print('Device set 2 optics path list:', optics_local_all_paths_2)

        print('HEREEEE', ec_local_all_paths)
        ec_local_all_paths = [Path(str(item_path).replace(' (7:24:24)', '')) for item_path in ec_local_all_paths]
        optics_local_all_paths = [Path(str(item_path).replace(' (7:24:24)', '')) for item_path in optics_local_all_paths]
        ec_local_all_paths_2 = [Path(str(item_path).replace(' (7:24:24)', '')) for item_path in ec_local_all_paths_2]
        optics_local_all_paths_2 = [Path(str(item_path).replace(' (7:24:24)', '')) for item_path in optics_local_all_paths_2]

        def find_matching_jpgs(unique_dirs, keywords):
            matching_files = []
            for directory in unique_dirs:
                for root, _, files in os.walk(directory):
                    for file in files:
                        if file.endswith('.jpg') and all(keyword in file.lower() for keyword in keywords):
                            item_path = os.path.join(root, file)
                            # item_path = Path(str(item_path).replace(' (7:24:24)', ''))
                            matching_files.append(item_path)
            return matching_files

        def find_matching_gifs(unique_dirs):
            matching_files = []
            for directory in unique_dirs:
                for root, _, files in os.walk(directory):
                    for file in files:
                        if file.endswith('.gif'):
                            item_path = os.path.join(root, file)
                            # item_path = Path(str(item_path).replace(' (7:24:24)', ''))
                            matching_files.append(item_path)
            return matching_files
        
        unique_dirs = []
        all_keyence_paths_1 = []
        all_keyence_paths_2 = []
        all_gif_paths_1 = []
        all_gif_paths_2 = []
        local_all_paths_1 = ec_local_all_paths
        local_all_paths_2 = ec_local_all_paths_2
        if local_all_paths_1: 
            # GET THE PARENT DIRECTORIES OF EACH DEVICE
            unique_dirs = extract_unique_parent_dirs(local_all_paths_1)
            #unique_dirs = ['/Users/sarahpearce/Library/CloudStorage/GoogleDrive-sarah@tynt.io/Shared drives/Data/Devices/2024/07/20240703/20240703_PB_4171/']
            if unique_dirs:
                for path in unique_dirs:
                    keywords = ['400x', 'mesh']  # Keywords to filter files
                    keyence_paths = find_matching_jpgs(unique_dirs, keywords)
                    all_keyence_paths_1.extend(keyence_paths)
                    gif_paths = find_matching_gifs(unique_dirs)
                    all_gif_paths_1.extend(gif_paths)

        if local_all_paths_2: 
            # GET THE PARENT DIRECTORIES OF EACH DEVICE
            unique_dirs = extract_unique_parent_dirs(local_all_paths_2)
            #unique_dirs = ['/Users/sarahpearce/Library/CloudStorage/GoogleDrive-sarah@tynt.io/Shared drives/Data/Devices/2024/07/20240703/20240703_PB_4171/']
            if unique_dirs:
                for path in unique_dirs:
                    keywords = ['400x', 'mesh']  # Keywords to filter files
                    keyence_paths = find_matching_jpgs(unique_dirs, keywords)
                    all_keyence_paths_2.extend(keyence_paths)
                    gif_paths = find_matching_gifs(unique_dirs)
                    all_gif_paths_2.extend(gif_paths)
            #print('Device Parent Directories Found:', unique_dirs)
            #print('Device Keyence paths found:', all_keyence_paths_1)

        ' ################### GETTING ACTUAL ARBIN DATA ##################### '
        arbin_df = get_devices_arbin_checkins(conn, cursor, device_id_list_1)
        if not arbin_df.empty: 
            single_cycles_df = get_devices_single_cycle_arbin_checkins(conn, cursor, arbin_df)
        else: 
            single_cycles_df = pd.DataFrame(data=[])
            print('NO DATA WARNING: No Arbin Data Uplaoded')

        ' ################### GETTING SINGLE VAL CHECKIN DATA ##################### '
        #checkin_df_dict = get_haze_weight_meshwidth_devicewidth_bubbles_ir_joined(conn, cursor, device_id_list_1)
        #print(checkin_df_dict)

        #unique_dirs = extract_unique_parent_dirs(local_all_paths)
        #print('unique_directories:', unique_dirs)



    ' ############################### MAKE ARBIN PLOTS ################################# '
    # Create all plots
    arbin_plots_layout = create_single_panel_plot(single_cycles_df) # handles empty dfs in function

    ' ############################### MAKE NON-CYCLE CHECKIN PLOTS ################################# '
    import matplotlib.pyplot as plt

    checkin_df_dict_1 = get_haze_weight_meshwidth_devicewidth_bubbles_ir_joined(conn, cursor, device_id_list_1)
    print(device_id_list_2)
    checkin_df_dict_2 = get_haze_weight_meshwidth_devicewidth_bubbles_ir_joined(conn, cursor, device_id_list_2)

    #ambient_noncycle_jmp_layout = comparison_create_static_noncycling_plot_dictionary_from_df(checkin_df_dict_1, selected_devices_1, checkin_df_dict_2, selected_devices_2, selected_name_1, selected_name_2)

    #ambient_noncycle_jmp_layout = comparison_noncycling_plots(checkin_df_dict_1, selected_devices_1, checkin_df_dict_2, selected_devices_2, selected_name_1, selected_name_2)
    plots = comparison_noncycling_plots(checkin_df_dict_1, selected_devices_1, checkin_df_dict_2, selected_devices_2, selected_name_1, selected_name_2)
    image_panes = [pn.pane.PNG(filename, width=400, height=400) for filename in plots.values()]
    ambient_noncycle_jmp_layout= pn.GridBox(*image_panes, ncols=2, sizing_mode='stretch_width')

    ' ############################### MAKE ECCHECKIN/OPTICSCHECKIN PLOTS ################################# '
    # Create all plots
    #jmp_plots_layout = create_single_panel_plot(single_cycles_df)
    ec_optics_df_1 = get_ec_optics_joined(conn, cursor, device_id_list_1)
    ec_optics_df_1['coulombic_efficiency'] *= 100
    ec_optics_df_1 = ec_optics_df_1.loc[:, ~ec_optics_df_1.columns.duplicated()]

    ec_optics_df_2 = get_ec_optics_joined(conn, cursor, device_id_list_2)
    ec_optics_df_2['coulombic_efficiency'] *= 100
    ec_optics_df_2 = ec_optics_df_2.loc[:, ~ec_optics_df_2.columns.duplicated()]
    jmp_plots_layout2 = create_jmp_panel(ec_optics_df_2)
    # save ec_optics_df to work on separately! 
    # Save ec_optics_df to a CSV file in the current folder
    '''#### WANT TO SAVE THE WITH ROUTE INFORMATION ABOVE AND HAVE A ROUTE DROPDOWN AS WELL'''

    # Non-interactive plots
    selected_devices_1 = selected_devices_1.rename(columns={'id': 'device_id'})
    selected_devices_2 = selected_devices_2.rename(columns={'id': 'device_id'})

    selected_devices_1['device_id'] = selected_devices_1['device_id'].astype(int)
    ec_optics_df_1['device_id'] = ec_optics_df_1['device_id'].astype(int)
    ec_optics_df_with_route_1 = pd.merge(selected_devices_1, ec_optics_df_1, on='device_id')
    ec_optics_df_1 = ec_optics_df_with_route_1
    ec_optics_df_1 = ec_optics_df_1.fillna(np.nan)

    selected_devices_2['device_id'] = selected_devices_2['device_id'].astype(int)
    ec_optics_df_2['device_id'] = ec_optics_df_2['device_id'].astype(int)
    ec_optics_df_with_route_2 = pd.merge(selected_devices_2, ec_optics_df_2, on='device_id')
    ec_optics_df_2 = ec_optics_df_with_route_2
    ec_optics_df_2 = ec_optics_df_2.fillna(np.nan)

    ec_optics_df = ec_optics_df_1 

    print('NA Filled df:', ec_optics_df)
    if ec_optics_df_1.empty:
        print('NO DATA WARNING: No JMP Data for set 1!')
    if ec_optics_df_2.empty:
        print('NO DATA WARNING: No JMP Data for set 2!')
    #ec_optics_df.to_csv('ec_optics_df.csv', index=False) # WILL USE WITH CREATE_JMP_PANEL

    " ####################### INTERACTIVE JMP THINGS (CYCLING) WITH SLIDER ############################## "
    # Note: within dataframe, this function checks (1) if dataframe is empty
    # then (2) for y variables with no data in the dataframe before plotting 
    interactive_slider_jmp_layout_1 = create_interactive_jmp_panel(ec_optics_df_1)
    interactive_slider_jmp_layout_2 = create_interactive_jmp_panel(ec_optics_df_2)

    " ########################## INTERACTIVE JMP THINGS (CYCLING) WITHOUT SLIDER ######################### "
    common_columns = ec_optics_df_1.columns.intersection(ec_optics_df_2.columns)
    if common_columns.empty:
        print("NO DATA WARNING: No common columns found between ec_optics_df_1 and ec_optics_df_2.")
        return  # or handle this case as needed
    # Filter both dataframes to only include common columns
    ec_optics_df_1_filtered = ec_optics_df_1[common_columns]
    ec_optics_df_2_filtered = ec_optics_df_2[common_columns]

    cycle_layout = comparison_create_static_cycling_jmp_panel2(ec_optics_df_1, ec_optics_df_2, selected_name_1, selected_name_2)


    " ############### DATA VIEWER BUILDING ################################################# "
    ec_file_list = [
    '/Users/sarahpearce/Library/CloudStorage/GoogleDrive-sarah@tynt.io/Shared drives/Data/Devices/2024/10/20241014/20241014_LS_4564/cycle1/20241014_LS_4564_cycle1.tyntEC',
    '/Users/sarahpearce/Library/CloudStorage/GoogleDrive-sarah@tynt.io/Shared drives/Data/Devices/2024/10/20241014/20241014_LS_4564/cycle6/20241014_LS_4564_cycle6.tyntEC'
    ]

    optics_file_list = [
        '/Users/sarahpearce/Library/CloudStorage/GoogleDrive-sarah@tynt.io/Shared drives/Data/Devices/2024/10/20241014/20241014_LS_4564/cycle1/20241014_LS_4564_cycle1.tyntOptics',
        '/Users/sarahpearce/Library/CloudStorage/GoogleDrive-sarah@tynt.io/Shared drives/Data/Devices/2024/10/20241014/20241014_LS_4564/cycle6/20241014_LS_4564_cycle6.tyntOptics'
    ]

    import holoviews as hv
    import hvplot.pandas

    # Enable Panel and HoloViews extensions
    hv.extension('bokeh')
    pn.extension()

    # Paths for files
    gif_file_list_1 = [
        '/Users/sarahpearce/Library/CloudStorage/GoogleDrive-sarah@tynt.io/Shared drives/Data/Devices/2024/10/20241014/20241014_LS_4565/cycle1/pictures/20241014_LS_4565_cycle1.gif'
    ]
    gif_file_list_2 = [
        '/Users/sarahpearce/Library/CloudStorage/GoogleDrive-sarah@tynt.io/Shared drives/Data/Devices/2024/10/20241014/20241014_LS_4565/cycle1/pictures/20241014_LS_4565_cycle1.gif'
    ]

    all_gif_paths_1  = [Path(str(item_path).replace(' (7:24:24)', '')) for item_path in all_gif_paths_1 ]
    all_gif_paths_2  = [Path(str(item_path).replace(' (7:24:24)', '')) for item_path in all_gif_paths_2 ]

    gif_file_list_1 = all_gif_paths_1 
    gif_file_list_2 = all_gif_paths_2
    optics_file_list_1 = optics_local_all_paths
    ec_file_list_1 = ec_local_all_paths
    optics_file_list_2 = optics_local_all_paths_2
    ec_file_list_2 = ec_local_all_paths_2

    # Helper function to create dropdown links with a default option
    def create_file_links(file_list):
        file_links = {"Select File": None}  # Default option
        for file_path in file_list:
            base_name = os.path.basename(file_path)
            name_without_ext = os.path.splitext(base_name)[0]
            file_links[name_without_ext] = file_path
        return file_links

    # Create dropdown links
    ec_file_links_1 = create_file_links(ec_file_list_1)
    optics_file_links_1 = create_file_links(optics_file_list_1)
    gif_file_links_1 = create_file_links(gif_file_list_1)

    ec_file_links_2 = create_file_links(ec_file_list_2)
    optics_file_links_2 = create_file_links(optics_file_list_2)
    gif_file_links_2 = create_file_links(gif_file_list_2)

    import hvplot.pandas
    import holoviews as hv
    # Enable Panel
    hv.extension('bokeh')
    pn.extension()

    # Dropdown widgets
    ec_dropdown_1 = pn.widgets.Select(name='EC Viewer 1', options=ec_file_links_1)
    ec_dropdown_2 = pn.widgets.Select(name='EC Viewer 2', options=ec_file_links_2)
    optics_dropdown_1 = pn.widgets.Select(name='Optics Viewer 1', options=optics_file_links_1)
    optics_dropdown_2 = pn.widgets.Select(name='Optics Viewer 2', options=optics_file_links_2)
    gif_dropdown_1 = pn.widgets.Select(name='GIF Viewer 1', options=gif_file_links_1)
    gif_dropdown_2 = pn.widgets.Select(name='GIF Viewer 2', options=gif_file_links_2)

    # Placeholders for viewers
    ec_plot_pane_1 = pn.pane.HoloViews(object=hv.Curve([]))
    ec_plot_pane_2 = pn.pane.HoloViews(object=hv.Curve([]))
    optics_plot_pane_1 = pn.pane.HoloViews(object=hv.Curve([]))
    optics_plot_pane_2 = pn.pane.HoloViews(object=hv.Curve([]))
    gif_pane_1 = pn.pane.Image(sizing_mode='stretch_both', embed=True)
    gif_pane_2 = pn.pane.Image(sizing_mode='stretch_both', embed=True)


    # Generalized callback functions
    def load_and_plot_ec(event, pane):
        file_path = event.new
        print('selected file', file_path)
        if not file_path:
            pane.object = hv.Text(0.5, 0.5, "No Data Selected").opts(width=800, height=400)
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
                width=800, height=400, legend_position='right'
            )

            pane.object = combined_plot
        except Exception as e:
            pane.object = hv.Text(0.5, 0.5, f"Error: {e}").opts(width=800, height=400)

    def load_and_plot_optics(event, pane):
        file_path = event.new
        print('selected optics file', file_path)
        if not file_path:
            pane.object = hv.Text(0.5, 0.5, "No Data Selected").opts(width=800, height=400)
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

            pane.object = transmission_plot
        except Exception as e:
            pane.object = hv.Text(0.5, 0.5, f"Error: {e}").opts(width=800, height=400)

    def display_gif(event, pane):
        file_path = event.new
        print('Selected GIF:', file_path)  # Debugging log
        if not file_path:
            pane.object = None  # Reset pane if no file selected
            return
        if not os.path.exists(file_path):
            pane.object = None
            print(f"File does not exist: {file_path}")
            return
        try:
            print(f"Setting GIF: {file_path}")
            pane.object = None  # Force refresh
            pane.object = file_path  # Update the pane with the new GIF
            print(f"Displaying GIF: {file_path}")
        except Exception as e:
            print(f"Error displaying GIF: {e}")
            pane.object = None


    # Link callbacks
    ec_dropdown_1.param.watch(lambda event: load_and_plot_ec(event, ec_plot_pane_1), 'value')
    ec_dropdown_2.param.watch(lambda event: load_and_plot_ec(event, ec_plot_pane_2), 'value')
    optics_dropdown_1.param.watch(lambda event: load_and_plot_optics(event, optics_plot_pane_1), 'value')
    optics_dropdown_2.param.watch(lambda event: load_and_plot_optics(event, optics_plot_pane_2), 'value')
    gif_dropdown_1.param.watch(lambda event: display_gif(event, gif_pane_1), 'value')
    gif_dropdown_2.param.watch(lambda event: display_gif(event, gif_pane_2), 'value')

    # Layouts
    ec_viewers = pn.Row(
        pn.Column("### EC Viewer 1", ec_dropdown_1, ec_plot_pane_1),
        pn.Column("### EC Viewer 2", ec_dropdown_2, ec_plot_pane_2)
    )

    optics_viewers = pn.Row(
        pn.Column("### Optics Viewer 1", optics_dropdown_1, optics_plot_pane_1),
        pn.Column("### Optics Viewer 2", optics_dropdown_2, optics_plot_pane_2)
    )

    gif_viewers = pn.Row(
        pn.Column("### GIF Viewer 1", gif_dropdown_1, gif_pane_1),
        pn.Column("### GIF Viewer 2", gif_dropdown_2, gif_pane_2)
    )

    # Combine sections
    data_viewer_content = pn.Column(ec_viewers, optics_viewers, gif_viewers)






    ' ############## KEYENCE SECTION ################### '

    # File lists for Experiment 1 and Experiment 2
    experiment_1_files = all_keyence_paths_1
    experiment_2_files = all_keyence_paths_2

    # Helper function to create dropdown links with a default option
    def create_file_links(file_list):
        file_links = {"Select File": None}  # Add default "Select File" option
        for file_path in file_list:
            base_name = os.path.basename(file_path)
            file_links[base_name] = file_path
        return file_links

    experiment_1_links = create_file_links(experiment_1_files)
    experiment_2_links = create_file_links(experiment_2_files)

    # Dropdown widgets
    experiment_1_dropdown = pn.widgets.Select(name='Experiment 1 Keyence Mesh Images', options=experiment_1_links)
    experiment_2_dropdown = pn.widgets.Select(name='Experiment 2 Keyence Mesh Images', options=experiment_2_links)

    # Placeholders for image display and file paths
    experiment_1_image_pane = pn.pane.PNG(sizing_mode='stretch_both')
    experiment_2_image_pane = pn.pane.PNG(sizing_mode='stretch_both')
    experiment_1_source_pane = pn.pane.Markdown(object="No Image Selected")
    experiment_2_source_pane = pn.pane.Markdown(object="No Image Selected")

    # Callback function for Experiment 1
    def update_experiment_1_image(event):
        file_path = event.new
        if not file_path:
            experiment_1_image_pane.object = None
            experiment_1_source_pane.object = "No Image Selected"
            return
        try:
            experiment_1_image_pane.object = file_path
            experiment_1_source_pane.object = f"**Source:** {file_path}"
        except Exception as e:
            experiment_1_image_pane.object = None
            experiment_1_source_pane.object = f"Error: {e}"

    # Callback function for Experiment 2
    def update_experiment_2_image(event):
        file_path = event.new
        if not file_path:
            experiment_2_image_pane.object = None
            experiment_2_source_pane.object = "No Image Selected"
            return
        try:
            experiment_2_image_pane.object = file_path
            experiment_2_source_pane.object = f"**Source:** {file_path}"
        except Exception as e:
            experiment_2_image_pane.object = None
            experiment_2_source_pane.object = f"Error: {e}"

    # Link callbacks to dropdowns
    experiment_1_dropdown.param.watch(update_experiment_1_image, 'value')
    experiment_2_dropdown.param.watch(update_experiment_2_image, 'value')

    # Layout for Experiment 1
    experiment_1_section = pn.Column(
        pn.pane.Markdown("### Experiment 1 Keyence Viewer"),
        experiment_1_dropdown,
        experiment_1_image_pane,
        experiment_1_source_pane
    )

    # Layout for Experiment 2
    experiment_2_section = pn.Column(
        pn.pane.Markdown("### Experiment 2 Keyence Viewer"),
        experiment_2_dropdown,
        experiment_2_image_pane,
        experiment_2_source_pane
    )

    # Combine sections in a Row
    keyence_viewer_content = pn.Row(experiment_1_section, experiment_2_section)




    '###### DASHBOARD ######### '
    
    logo_path = os.path.join(os.getcwd(), 'figures', 'tynt_logo.png')

    # main content area
    main_content = pn.Column(pn.Column(''## All Devices Selected:', 
                         '### Device Set 1 from ' + selected_name_1,
                         pn.pane.HTML(table_html_1), 
                         '### Device Set 2 from ' + selected_name_2,
                         pn.pane.HTML(table_html_2),)
    )

    section1 = pn.Column('## All Devices Selected:', 
                         '### Device Set 1 from ' + selected_name_1,
                         pn.pane.HTML(table_html_1), 
                         '### Device Set 2 from ' + selected_name_2,
                         pn.pane.HTML(table_html_2),
                        pn.pane.HTML(report_html))
    
    section2 = pn.Column('## Checkin Summary Values from Database', 
                         #pn.Row(pn.Column('## Interactive Plots:'), pn.Column(jmp_plots_layout)),
                         pn.Column(pn.Row('### Non-cycling Checkins:'), 
                                pn.Row(ambient_noncycle_jmp_layout)),
                        pn.Column(pn.Row('### Cycing Checkins:'), 
                                                    pn.Row(cycle_layout)))

    section3 = pn.Column('## Checkin Summary Values from Database', 
                        #pn.Row(pn.Column('## Interactive Plots:'), pn.Column(jmp_plots_layout)),
                        pn.Column(pn.Row('### Non-cycling Checkins:'), 
                            pn.Row('# Developing')),
                    pn.Column(pn.Row('### Cycing Checkins:'), 
                                                pn.Row('# Developing')))
    
    section4 = pn.Column('## Raw Data Viewers for Optics/Photo CheckIns', pn.Column(data_viewer_content))
    section5 = pn.Column('## Keyence Images', pn.Column(keyence_viewer_content))

    main_content = pn.Tabs(
        ('Devices', section1),
        ('Ambient Route CheckIn Performance', section2),
        ('Weatherometer Route CheckIn Performance', section3),
        ('Raw Data Viewers', section4),
        ('Keyence Images', section5),
    )
    
        # Define custom CSS for 3D effect

    # Create buttons in the sidebar to navigate to each section
    button1 = pn.widgets.Button(name='Go to Device Details', button_type='primary')
    button2 = pn.widgets.Button(name='Go to Summarized Checkin Cycling Data (Ambient)', button_type='primary')
    button3 = pn.widgets.Button(name='Go to Summarized Checkin Cycling Data (Weatherometer)', button_type='primary')
    button4 = pn.widgets.Button(name='Go to Raw Data Viewers', button_type='primary')
    button5 = pn.widgets.Button(name='Go to Keyence Images', button_type='primary')
    # Define callback functions for buttons
    def go_to_section1(event):
        main_content.active = 0
    def go_to_section2(event):
        main_content.active = 1
    def go_to_section3(event):
        main_content.active = 2
    def go_to_section4(event):
        main_content.active = 3
    def go_to_section5(event):
        main_content.active = 4
    # Attach callbacks to buttons
    button1.on_click(go_to_section1)
    button2.on_click(go_to_section2)
    button3.on_click(go_to_section3)
    button4.on_click(go_to_section4)
    button5.on_click(go_to_section5)

    # Define the content for the sidebar
    sidebar = pn.Column(
        pn.pane.PNG(logo_path, width=150, height=100),
            pn.pane.Markdown("### Description of Runs: "),
            pn.pane.Markdown('#### Experiment Name 1: ' + selected_name_1), 
            pn.pane.Markdown("Database Notes: " + notes_string_1), 
            pn.pane.Markdown('#### Experiment Name 2: ' + selected_name_2), 
            pn.pane.Markdown("Database Notes: " + notes_string_2), 

            pn.Column(button1, button2, button3, button4, button5)
    )

    template = pn.template.FastListTemplate(
        title='Experiment Comparison and Reporting Dashboard',
        sidebar=sidebar,
        main=main_content,
        accent_base_color="#00564a",
        header_background="#00564a",
    )
    template.show()
    

if __name__ == "__main__":
    main()

