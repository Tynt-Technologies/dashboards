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
from scripts.html_functions import *
import panel as pn
from PIL import Image
import os

# Sarah's mac:  source dashboard_env/bin/activate  


hv.extension('bokeh')
pn.extension()

" ################### TRD SPECIFIC PANEL FUNCTIONS ########################## "

" ######################################## START GUIS ########################################################## "
def create_dynamic_dialog(trds_df, conn, cursor):
    # Create the application and the main window
    app = QApplication([])
    window = QWidget()
    window.setWindowTitle("Dynamic Dropdown Example")

    # Create a layout for the main window
    main_layout = QVBoxLayout(window)

    # Label and first dropdown
    label = QLabel("Select a TRD Name:")
    main_layout.addWidget(label)

    first_dropdown = QComboBox()
    first_dropdown.addItems(reversed(trds_df['trd_name'].values)) 
    main_layout.addWidget(first_dropdown)

    # Create a scroll area for checkboxes
    scroll_area = QScrollArea()
    checkbox_widget = QWidget()
    checkbox_layout = QVBoxLayout(checkbox_widget)
    checkbox_widget.setLayout(checkbox_layout)
    scroll_area.setWidget(checkbox_widget)
    scroll_area.setWidgetResizable(True)
    main_layout.addWidget(scroll_area)

    # Function to update checkboxes based on the first dropdown selection
    def update_checkboxes():
        # Clear previous checkboxes
        for i in reversed(range(checkbox_layout.count())):
            checkbox_layout.itemAt(i).widget().setParent(None)
        
        category = first_dropdown.currentText()
        matching_ids = search_trd_name(trds_df, category)
        trd_id = matching_ids[0]
        trd_devices = get_trd_devices(conn, cursor, trd_id)
        device_list = trd_devices['device_name'].values  # Assuming trd_devices has 'device_name' column

        for device in device_list:
            checkbox = QCheckBox(device)
            checkbox.setChecked(True)
            checkbox_layout.addWidget(checkbox)

    # Connect the first dropdown's change event to the update function
    first_dropdown.currentTextChanged.connect(update_checkboxes)

    # Initialize the checkboxes for the first time
    update_checkboxes()

    # Variables to hold the selection
    selected_trd_name = None
    selected_devices = []

    # Function to capture the selections
    def capture_selections():
        nonlocal selected_trd_name, selected_devices
        selected_trd_name = first_dropdown.currentText()
        selected_devices = [checkbox.text() for checkbox in checkbox_widget.findChildren(QCheckBox) if checkbox.isChecked()]
        print(f"Selected TRD Name: {selected_trd_name}")
        print(f"Selected Devices: {selected_devices}")
        app.quit()  # Close the application

    # Create a button to confirm selection
    button = QPushButton("OK")
    button.clicked.connect(capture_selections)
    main_layout.addWidget(button)

    # Adjust the window size based on the contents
    window.adjustSize()

    # Show the window and execute the application
    window.show()
    app.exec()

    # Return the selections after the window is closed
    return selected_trd_name, selected_devices

def get_user_input(options):
    """
    Create a dropdown menu for user selection using PySide6 and return the selected option.

    Parameters:
    - options: list of strings to display in the dropdown menu

    Returns:
    - The selected option as a string
    """
    app = QApplication([])

    # Create the main window
    window = QWidget()
    window.setWindowTitle('Select an Option')

    # Create a vertical layout
    layout = QVBoxLayout()

    # Create a label
    label = QLabel('Please select an option:')
    layout.addWidget(label)

    # Create a dropdown menu (QComboBox)
    dropdown = QComboBox()
    dropdown.addItems(options)
    layout.addWidget(dropdown)

    # Create a button to confirm selection
    button = QPushButton('OK')
    layout.addWidget(button)

    # Create a label to display the selected option
    result_label = QLabel('')
    layout.addWidget(result_label)

    # Variable to store the selected option
    selected_option = [None]

    # Function to handle button click
    def on_button_click():
        selected_option[0] = dropdown.currentText()
        result_label.setText(f"You selected: {selected_option[0]}")
        print(f"You selected: {selected_option[0]}")
        app.quit()

    button.clicked.connect(on_button_click)

    # Set the layout and show the window
    window.setLayout(layout)
    window.show()

    # Execute the application and wait for the event loop to quit
    app.exec()

    return selected_option[0]

from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QListWidget, QPushButton, QListWidgetItem

def get_user_selected_cycles(cycles):
    app = QApplication([])

    # Create the main window
    window = QWidget()
    window.setWindowTitle("Select Cycles")
    layout = QVBoxLayout()

    # Create a QListWidget
    list_widget = QListWidget()

    # Add cycles to the QListWidget with checkboxes
    for cycle in cycles:
        item = QListWidgetItem(str(cycle))
        item.setCheckState(Qt.Checked)
        list_widget.addItem(item)

    # Add the QListWidget to the layout
    layout.addWidget(list_widget)

    # Create and add a button to confirm selection
    button = QPushButton("OK")
    layout.addWidget(button)

    # Set the layout to the window
    window.setLayout(layout)

    # Function to close the window and process selected cycles
    def confirm_selection():
        selected_cycles = []
        for index in range(list_widget.count()):
            item = list_widget.item(index)
            if item.checkState() == Qt.Checked:
                selected_cycles.append(int(item.text()))
        window.close()
        return selected_cycles

    # Connect the button to the function
    button.clicked.connect(confirm_selection)

    # Show the window and execute the app
    window.show()
    app.exec()

    return confirm_selection()


" ######################################## END GUIS ########################################################## "


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

def get_wellplates(conn, cursor):
    if conn and cursor:
        sql_query = '''
            SELECT * 
            FROM tyntdatabase_wellplate 
            LIMIT ALL 
            OFFSET 0;
        '''
        cursor.execute(sql_query)
        wellplates = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        wellplates = pd.DataFrame(wellplates, columns=column_names)

    return wellplates

def get_wellplate_CAs(conn, cursor):
    if conn and cursor:
        sql_query = '''
            SELECT * 
            FROM tyntdatabase_wellplatecacheckin
            LIMIT ALL 
            OFFSET 0;
        '''
        cursor.execute(sql_query)
        wellplatecas = cursor.fetchall()
            # Fetch the column names
        column_names = [desc[0] for desc in cursor.description]
        wellplatecas = pd.DataFrame(wellplatecas, columns=column_names)
        # print("Column names:", column_names)
    return wellplatecas

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
    selected_name, selected_devices, experiment_type = create_dynamic_baseline_and_trd_dialog(all_baselines_df, all_trds_df, conn, cursor)
    print(f"Selected Experiment Version: {selected_name}")
    print(f"Selected Devices: {selected_devices}")

    search_string = selected_name

    if search_string and selected_devices:
        print(experiment_type)
        print(f"Search String Entered: {search_string}")
        if experiment_type == 'Baseline':
            print('Identified as Basline')
            # Get notes
            sub_df = all_baselines_df[all_baselines_df['baseline_version'] == search_string]
            notes_string = sub_df['notes'].values[0]
            # Get ID and Devices
            matching_ids = search_baseline_name(all_baselines_df, search_string)
            baseline_id = matching_ids[0]
            baseline_devices = get_baseline_devices(conn, cursor, baseline_id)
            # Get list of IDs for the specified device names
            device_id_list = baseline_devices.loc[baseline_devices['device_name'].isin(selected_devices), 'id'].tolist()
            print('IDs corresponding to selected devices: ', device_id_list)
            device_list = selected_devices
            print('device list: ', device_id_list)
            baseline_eccheckins = get_trd_eccheckins(conn, cursor, device_id_list) # function works for either
            path_list = baseline_eccheckins['server_path'].values 

        elif experiment_type == 'TRD':
            print('Identified as TRD')
                    # Get notes
            sub_df = all_trds_df[all_trds_df['trd_name'] == search_string]
            notes_string = sub_df['notes'].values[0]
            # Get ID and Devices
            matching_ids = search_trd_name(all_trds_df, search_string)
            trd_id = matching_ids[0]
            trd_devices = get_trd_devices(conn, cursor, trd_id)
            # Get list of IDs for the specified device names
            device_id_list = trd_devices.loc[trd_devices['device_name'].isin(selected_devices), 'id'].tolist()
            print('IDs corresponding to selected devices: ', device_id_list)
            device_list = selected_devices
            print('device list: ', device_id_list)
            # Get paths
            trd_eccheckins = get_trd_eccheckins(conn, cursor, device_id_list)
            path_list = trd_eccheckins['server_path'].values 

        current_directory = os.getcwd()
        # cv_image_path = os.path.join(current_directory, '/figures/no_initial_photo_available.jpg')
        # no_data_image_path = os.path.join(current_directory, '/figures/no_data.png')

        
        path_list = [item for item in path_list if item is not None]
        print('old ALL path list: ', path_list)
        local_all_paths = []
        local_all_paths = get_local_paths(path_list)
        print('new ALL path list:', local_all_paths)

        ' ################################ BUILD ENTIRE DASHBOARD AS EMPTY FIRST #################################### '

        ' ###### SIDEBAR ###### '

        '#### PAGE 1: EMPTIES ####### '
        keyence_paths = ['None Found']
        local_warmup_paths = ['None Found']
        optics_folder_paths = ['In Progress']
        photo_folder_paths = ['None Found']
        arbin_paths = ['None Found']

        '#### PAGE 2: EMPTIES ####### '
        warmup_path_list = []
        local_warmup_paths = []
        warmup_folder_paths = []
        warmup_ecs_corresponding_to_photos = []
        cycling_folder_paths = []
        warmup_folder_paths = []
        all_folders_list = []
        photo_folder_paths = []

        '#### PAGE 3: EMPTIES ####### '
        '#### PAGE 4: EMPTIES ####### '
        '#### PAGE 5: EMPTIES ####### '

        '#### PAGE 1: GET DATA ####### '

        ' ############################# CALLING ALL HTML FUNCTIONS FOR DEVICE OVERVIEW TAB ################### '
        # adding route_name to baseline devices dataframe
        routes_df = get_routes(conn, cursor)
        if experiment_type == 'Baseline':
            baseline_devices = get_baseline_devices(conn, cursor, baseline_id)
            baseline_devices['route_id'] = [0 if x is None else x for x in baseline_devices['route_id'].values]
            routes_df = pd.concat([new_row, routes_df], ignore_index=True)
            baseline_devices = pd.merge(baseline_devices, routes_df, on='route_id')
            print('FINAL TABLE OF DEVICE DATA', baseline_devices.columns)
            table_html = all_devices_table(baseline_devices)
            selected_devices = baseline_devices

        elif experiment_type == 'TRD':
            trd_devices = get_trd_devices(conn, cursor, trd_id)
            print('trddf',trd_devices)
            print('routedf',routes_df)
            print(trd_devices['route_id'].values)
            # Handle unspecified case
            trd_devices['route_id'] = [0 if x is None else x for x in trd_devices['route_id'].values]
            new_row = pd.DataFrame({'route_id': [0], 'route_name': ['Unspecified']})
            routes_df = pd.concat([new_row, routes_df], ignore_index=True)
            # Continue merging
            trd_devices = pd.merge(trd_devices, routes_df, on='route_id')
            print('FINAL TABLE OF DEVICE DATA', trd_devices.columns)
            table_html = all_devices_table(trd_devices)
            selected_devices = trd_devices

        # Bullet list of gathered data
        warmups = local_warmup_paths
        o_checks = optics_folder_paths
        p_checks = photo_folder_paths
        arbin = arbin_paths
        keyence = keyence_paths
        report_html = generate_devices_report(warmups, o_checks, p_checks, arbin, keyence)

        '#### PAGE 2: GET DATA ####### '
        unique_dirs = []
        arbin_paths = []
        if local_all_paths: 
            # GET THE PARENT DIRECTORIES OF EACH DEVICE
            unique_dirs = extract_unique_parent_dirs(local_all_paths)
            #unique_dirs = ['/Users/sarahpearce/Library/CloudStorage/GoogleDrive-sarah@tynt.io/Shared drives/Data/Devices/2024/07/20240703/20240703_PB_4171/']
            if unique_dirs:
                for path in unique_dirs:
                    arbin_path = get_all_arbin_folders(path) 
                    arbin_paths.append(arbin_path)
            print('Device Parent Directories Found:', unique_dirs)
            print('Device Arbin paths found:', arbin_paths)

        warmups_df = get_baseline_warmups(conn, cursor, device_id_list) # works for either
        if not warmups_df.empty: 
            warmup_path_list = warmups_df['server_path'].values 
            warmup_path_list = [item for item in warmup_path_list if item is not None]
            print('old warmup path list: ', warmup_path_list)
            local_warmup_paths = get_local_paths(warmup_path_list)
            print('local warmup path list: ', local_warmup_paths) 

        # GET THE WARMUP PHOTO PATHS
        if len(unique_dirs) > 0:
            test = unique_dirs[0]

            all_folders_list = find_folders_recursively(test)
            photo_folder_paths = find_photo_folder_paths(all_folders_list)
            
            warmup_folder_paths = find_warmup_paths(photo_folder_paths)
            print('Warmup Photo Folders:', warmup_folder_paths)
            warmup_gif_paths = []
            for warmup_folder_path in warmup_folder_paths:
                gif_path = get_gif_path(warmup_folder_path) # returns '' if no gif found
                warmup_gif_paths.append(gif_path)
            print('Warmup Folder GIF files:', warmup_gif_paths)
            warmup_ecs_corresponding_to_photos = get_corresp_ec_filepaths(warmup_folder_paths)
            print('Warmup Folder EC files:', warmup_ecs_corresponding_to_photos)


            cycling_folder_paths = find_cycle_paths(photo_folder_paths)
            print('Cycling Photo Folders:', cycling_folder_paths)
            cycling_gif_paths = []
            for cycling_folder_path in cycling_folder_paths:
                gif_path = get_gif_path(cycling_folder_path) # returns '' if no gif found
                cycling_gif_paths.append(gif_path)
            print('Cycling Folder GIF files:', cycling_gif_paths)
            cycling_ecs_corresponding_to_photos = get_corresp_ec_filepaths(cycling_folder_paths)
            print('Cycling Folder EC files:', cycling_ecs_corresponding_to_photos)



        # final_df = get_all_raw_data(local_all_paths) # local all paths includes only things uploaded to db!!

        ' ################### GETTING ACTUAL ARBIN DATA ##################### '
        arbin_df = get_devices_arbin_checkins(conn, cursor, device_id_list)
        if not arbin_df.empty: 
            single_cycles_df = get_devices_single_cycle_arbin_checkins(conn, cursor, arbin_df)
        else: 
            single_cycles_df = pd.DataFrame(data=[])
            print('NO DATA WARNING: No Arbin Data Uplaoded')

        ' ################### GETTING SINGLE VAL CHECKIN DATA ##################### '
        checkin_df_dict = get_haze_weight_meshwidth_devicewidth_bubbles_ir_joined(conn, cursor, device_id_list)
        print(checkin_df_dict)

        unique_dirs = extract_unique_parent_dirs(local_all_paths)
        print('unique_directories:', unique_dirs)
            # For warmups, define the photo directories 
        # If no photos in unique_dirs


        # First set images paths to unavailable image/image not found
        image_paths = [os.path.join(os.getcwd(), 'figures', 'no_data.png'), os.path.join(os.getcwd(), 'figures', 'no_data.png')]
        ' ################## PICTURES ################ '

        # now want to use warmup_folder_paths and warmup_ecs_corresponding_to_photos 
        # need as many rows as have paths!!!
        # EDITING HERE
        #warmup_photo_rows = len(warmup_folder_paths)
        #photo_checkin_crop_box = (450, 1200, 2400, 3300)
        #image_paths = [os.path.join(warmup_folder_paths[0], fname) for fname in sorted(os.listdir(warmup_folder_paths[0])) if fname.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        #print('FINAL PATHS', image_paths)
        #image_panes = [pn.pane.Image(crop_image(path, photo_checkin_crop_box), width=400, height=300) for path in image_paths]


        # DIRECTORY WHERE IMAGES ARE STORED
        # SET TO NONE FIRST
        formatted_schedule = ''
        combined_plot = ''
        photo_step_descriptions = ['No Cycling Steps Found', 'No Cycling Steps Found']
        image_pane =  'No Pre-Cycling Photo Checkin GIFs Found' 
        schedule_and_plot_pane = 'No Pre-Cycling EC Files Found'

        # Define function to create panes for cycling and warmup files
        def impane_and_schedule(file_path, formatted_schedule, combined_plot, efficiency_text, gif_path, photo_step_descriptions):
            schedule_and_plot_pane = pn.Column(
                pn.pane.Markdown('### File: \n' + str(file_path)),
                pn.pane.Markdown('### Schedule: \n' + formatted_schedule),
                pn.pane.Markdown('### Corresponding EC file:'),
                combined_plot, efficiency_text)
            if gif_path: 
                file = gif_path.split('/')[-1]
                device_title = file.split('.')[0]
                gif_pane = pn.pane.Image(gif_path, width=600, height=400, align='start')
                image_pane = pn.Column(
                    pn.pane.Markdown(str(device_title)), gif_pane)
            else: 
                image_pane = pn.pane.Markdown('### No Photo Checkin GIF Found')
            return image_pane, schedule_and_plot_pane 
        #image_dir ='/Users/sarahpearce/Library/CloudStorage/GoogleDrive-sarah@tynt.io/Shared drives/Data/Devices/2024/07/20240703/20240703_PB_4172/precycle1/pictures'
        if warmup_folder_paths is not None and warmup_folder_paths:
            image_dir = warmup_folder_paths[0]

            file_path = warmup_ecs_corresponding_to_photos[0]
            gif_path = warmup_gif_paths[0]
            photo_step_descriptions = photo_step_description(file_path)
                
            # Get full schedule and plot EC curve with the photos FOR WARMUP
            formatted_schedule, combined_plot, efficiency_text = plot_data_and_print_schedule(file_path)
            image_pane, schedule_and_plot_pane = impane_and_schedule(file_path, formatted_schedule, combined_plot, efficiency_text, gif_path, photo_step_descriptions)
            

        # FOR CYCLING PHOTOS 
        formatted_schedule = ''
        combined_plot = ''
        photo_step_descriptions = ['No Cycling Steps Found', 'No Cycling Steps Found']
        c_image_pane =  'No Cycling Photo Checkin GIFs Found' 
        c_schedule_and_plot_pane = 'No Cycling EC Files Found'
        
        cycling_photo_rows = ''
        if cycling_folder_paths is not None and cycling_folder_paths:
            rows = []
            for i in range(len(cycling_ecs_corresponding_to_photos)):
                c_file_path = cycling_ecs_corresponding_to_photos[i]
                c_gif_path = cycling_gif_paths[i]
                c_photo_step_descriptions = photo_step_description(c_file_path)
                c_formatted_schedule, c_combined_plot, c_efficiency_text = plot_data_and_print_schedule(c_file_path)
                c_image_pane, c_schedule_and_plot_pane = impane_and_schedule(c_file_path, c_formatted_schedule, c_combined_plot, c_efficiency_text, c_gif_path, c_photo_step_descriptions)
                row_i = pn.Row(c_image_pane, c_schedule_and_plot_pane)
                rows.append(row_i)
            cycling_photo_rows = pn.Column(*rows)
        
            #row_names = cycling_gif_paths
            #row_names = []
            #for path in cycling_gif_paths:
            #    print(path)
            #    path = str(path).split('/')[-1]
            #    print(path)
            #    row_names.append(path)
            #row_dict = {name: row for name, row in zip(row_names, rows)}
            #def update_display(selected_row_name):
            #    return row_dict[selected_row_name]
            #dropdown = pn.widgets.Select(name='Select Device/Cycle', options=row_names)
            #selected_row = pn.bind(update_display, selected_row_name=dropdown.param.value)
            #cycling_photo_rows = pn.Column(dropdown, selected_row)



    ' ############################### MAKE ARBIN PLOTS ################################# '
    # Create all plots
    arbin_plots_layout = create_single_panel_plot(single_cycles_df) # handles empty dfs in function

    ' ############################### MAKE NON-CYCLE CHECKIN PLOTS ################################# '
    import matplotlib.pyplot as plt


    plots = create_static_noncycling_plot_dictionary_from_df(checkin_df_dict, selected_devices)

    #print('ALL PLOTS', plots)


    image_panes = [pn.pane.PNG(filename, width=400, height=400) for filename in plots.values()]
    noncycle_jmp_layout = pn.GridBox(*image_panes, ncols=2, sizing_mode='stretch_width')


    ' ############################### MAKE ECCHECKIN/OPTICSCHECKIN PLOTS ################################# '
    # Create all plots
    #jmp_plots_layout = create_single_panel_plot(single_cycles_df)
    ec_optics_df = get_ec_optics_joined(conn, cursor, device_id_list)
    ec_optics_df['coulombic_efficiency'] *= 100
    print('Original JMP dataframe:', ec_optics_df)
    ec_optics_df = ec_optics_df.loc[:, ~ec_optics_df.columns.duplicated()]
    jmp_plots_layout = create_jmp_panel(ec_optics_df)
    # save ec_optics_df to work on separately! 
    # Save ec_optics_df to a CSV file in the current folder
    '''#### WANT TO SAVE THE WITH ROUTE INFORMATION ABOVE AND HAVE A ROUTE DROPDOWN AS WELL'''

    # Non-interactive plots
    print('cycling dataframe columns:', ec_optics_df.columns)
    selected_devices = selected_devices.rename(columns={'id': 'device_id'})

    selected_devices['device_id'] = selected_devices['device_id'].astype(int)
    ec_optics_df['device_id'] = ec_optics_df['device_id'].astype(int)
    ec_optics_df_with_route = pd.merge(selected_devices, ec_optics_df, on='device_id')
    ec_optics_df = ec_optics_df_with_route

    ec_optics_df = ec_optics_df.fillna(np.nan)
    print('NA Filled df:', ec_optics_df)
    if ec_optics_df.empty:
        print('NO DATA WARNING: No JMP Data!')
    ec_optics_df.to_csv('ec_optics_df.csv', index=False) # WILL USE WITH CREATE_JMP_PANEL

    " ####################### INTERACTIVE JMP THINGS (CYCLING) WITH SLIDER ############################## "
    # Note: within dataframe, this function checks (1) if dataframe is empty
    # then (2) for y variables with no data in the dataframe before plotting 
    interactive_slider_jmp_layout = create_interactive_jmp_panel(ec_optics_df)

    " ########################## INTERACTIVE JMP THINGS (CYCLING) WITHOUT SLIDER ######################### "
    cycle_jmp_layout = create_static_cycling_jmp_panel(ec_optics_df)
    # cycle_jmp_layout = create_static_jmp_panel(ec_optics_df)
    # static_cycle_ jmp_layout = create_static_jmp_panel(ec_optics_df)

    " ########################### MAKE PREDICTIVE HEATMAP ############################## "
    print(ec_optics_df.columns)
    # Function to create heatmap
    # prep data 
    cycle2 = ec_optics_df[ec_optics_df['cycle_number'] == 2]
    'coulombic_efficiency', 'initial_percentage'
    checkin_df_dict # dataframe
    for name, df in checkin_df_dict.items():
        selected_devices = selected_devices.rename(columns={'id': 'device_id'})
        df_with_route = pd.merge(selected_devices, df, on='device_id')
        df = df_with_route
        if name == 'df_hazecheckin':
            x = df['check_in_age']
            y = df['check_in_haze']
            age_units = df['check_in_age_unit'].astype(str)  # Convert to string if needed
            route_names = df['route_name']  # Assuming this is the column with route names
    
    # device_id in both 

    #predictive_layout = correlation_heatmap(ec_optics_df)
    predictive_layout = correlation_heatmap_with_slope(ec_optics_df)



    
    '###### DASHBOARD ######### '
    
    logo_path = os.path.join(os.getcwd(), 'figures', 'tynt_logo.png')
    # image_pane, schedule_and_plot_pane
    # cycling_image_pane, cycling_schedule_and_plot_pane
    # Package details of the check-in schedule

    # REMOVE BUTTONS ON EITHER SIDE OF SLIDER 
    # Define button callbacks to adjust slider value
    #def increment_slider(event):
    #    slider.value = min(slider.end, slider.value + 1)
    #def decrement_slider(event):
    #    slider.value = max(slider.start, slider.value - 1)
    # Create functional arrow buttons
    #increment_button = pn.widgets.Button(name="▶", button_type="primary")
    #decrement_button = pn.widgets.Button(name="◀", button_type="primary")
    # Attach callbacks to buttons
    #increment_button.on_click(increment_slider)
    #decrement_button.on_click(decrement_slider)


    # Define the main content area
    main_content = pn.Column(pn.Column('## All Devices in Baseline Run:', pn.pane.HTML(table_html),
        pn.Column('## Warmup Data', pn.Row(image_pane, schedule_and_plot_pane),),
        pn.Column('## Cycling Data',))
    )

    section1 = pn.Column('## All Devices in Baseline Run:', pn.pane.HTML(table_html), pn.pane.HTML(report_html))
    section2 = pn.Column('## Warmup Data', pn.Row(image_pane, schedule_and_plot_pane),)
    section3 = pn.Column('## Cycling Data', cycling_photo_rows,)
    section4 = pn.Column('## Checkin Summary Values from Database', 
                         #pn.Row(pn.Column('## Interactive Plots:'), pn.Column(jmp_plots_layout)),
                         pn.Row(pn.Row('## Static Plots (Non-cycle Checkins):'), 
                                pn.Row(noncycle_jmp_layout)),
                        pn.Row(pn.Row('## Static Plots (Cycing Checkins):'), 
                                                    pn.Row(cycle_jmp_layout)))
    section5 = pn.Column('## Interactive Failure Analysis', pn.Column(interactive_slider_jmp_layout))
    section6 = pn.Column('## Arbin Summary Values from Database', arbin_plots_layout)
    section7 = pn.Column('## Keyence Images',)
    section8 = pn.Column('## Durability Predictions (keras/tensorflow modeling)', pn.Row(predictive_layout))

    main_content = pn.Tabs(
        ('Devices', section1),
        ('Warmup Data', section2),
        ('Cycling Data', section3),
        ('JMP Summary Plots', section4),
        ('Failure Analysis', section5),
        ('Arbin Summary Plots', section6),
        ('Keyence Images', section7),
        ('Predictive Modeling', section8),
    )
    
        # Define custom CSS for 3D effect

    # Create buttons in the sidebar to navigate to each section
    button1 = pn.widgets.Button(name='Go to Device Details', button_type='primary')
    button2 = pn.widgets.Button(name='Go to Warmup Data', button_type='primary')
    button3 = pn.widgets.Button(name='Go to Raw Cycling Data', button_type='primary')
    button4 = pn.widgets.Button(name='Go to Summarized Checkin Cycling Data', button_type='primary')
    button5 = pn.widgets.Button(name='Go to Interactive Failure Analysis', button_type='primary')
    button6 = pn.widgets.Button(name='Go to Summarized Arbin Cycling Data', button_type='primary')
    button7 = pn.widgets.Button(name='Go to Durability Predictions', button_type='primary')
    button8 = pn.widgets.Button(name='Go to Keyence Images', button_type='primary')
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
    def go_to_section6(event):
        main_content.active = 5
    def go_to_section7(event):
        main_content.active = 6
    def go_to_section8(event):
        main_content.active = 7
    # Attach callbacks to buttons
    button1.on_click(go_to_section1)
    button2.on_click(go_to_section2)
    button3.on_click(go_to_section3)
    button4.on_click(go_to_section4)
    button5.on_click(go_to_section5)
    button6.on_click(go_to_section6)
    button7.on_click(go_to_section7)
    button8.on_click(go_to_section8)

    # Define the content for the sidebar
    sidebar = pn.Column(
        pn.pane.PNG(logo_path, width=150, height=100),
            pn.pane.Markdown("### Description of Baseline Run: "),
            pn.pane.Markdown('Name: ' + search_string), 
            pn.pane.Markdown("Database Notes: " + notes_string),  
            pn.Column(button1, button2, button3, button4, button5, button6, button7)
    )

    template = pn.template.FastListTemplate(
        title='Baseline Reporting Dashboard',
        sidebar=sidebar,
        main=main_content,
        accent_base_color="#00564a",
        header_background="#00564a",
    )
    template.show()
    

if __name__ == "__main__":
    main()

