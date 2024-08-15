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

import platform
def search_shared_drives_data():
    # Define potential base paths to search (customize based on your setup)
    system = platform.system()
    if system == 'Windows': # is pc
        base_paths = [
        os.path.expanduser('~'),     # Home directory
        'C:\\',                      # Root of C: drive
        'D:\\',                      # Root of D: drive, adjust if you have other drives
        'E:\\'                       # Root of E: drive, adjust if needed
        ]
        
    elif system == 'Darwin': # is mac
        base_paths = [
        os.path.expanduser('~'),  # Home directory
        '/Volumes',               # Common mount points
        '/media',                 # Alternative common mount points
        '/mnt'                    # Alternative common mount points
        ]
    else:
        raise OSError("Unsupported operating system")
    
    target_directory = 'Shared drives'
    sub_directory = 'Data'

    for base_path in base_paths:
        path = find_directory(base_path, target_directory)
        if path:
            # Check within the found directory for the sub-directory 'Data'
            data_path = find_directory(path, sub_directory)
            if data_path:
                return data_path

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
    path_obj = Path(path)
    # Convert the path to a string and find the position of the substring
    path_str = str(path_obj)
    
    index = path_str.find(substring)
    if index == -1:
        return None  # Substring not found
    
    # Extract everything after the substring and handle leading slashes
    result = path_str[index + len(substring):].lstrip('/')
    
    return result

def combine_paths(base_path, relative_path):   
    # Fix google drive's duplicate issue?
    # Regular expression pattern to match a timestamp (e.g., (7:24:24))
    # base_path = '/Users/sarahpearce/Library/CloudStorage/GoogleDrive-sarah@tynt.io/Shared drives/Data'
    print(base_path)
    base_path_obj = Path(base_path)
    relative_path_obj = Path(relative_path)
    
    # Combine the paths
    final_path = base_path_obj / relative_path_obj
    
    return str(final_path)

def check_path_exists(path):
    """
    Check if the given path exists.

    Args:
        path (str): The path to check.

    Returns:
        bool: True if the path exists, False otherwise.
    """
    path_obj = Path(path)
    return path_obj.exists()

def get_local_paths(path_list):
    local_paths = []
    for path in path_list:
        substring = 'google-drive-data/'
        end_of_path = extract_after_substring(path, substring)
        print("Extracted Path:", end_of_path)

        # Find the 'Shared drives/Data' directory
        directory_path = search_shared_drives_data()
        print("Path to 'Shared drives/Data':", directory_path)

        final_path = combine_paths(directory_path, end_of_path)

        print("Final Path:", final_path)

        ## GETTING STUCK HERE????

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

    # category = 'OBL3-Contender2'
    selected_baseline_name, selected_devices = create_dynamic_baseline_dialog(all_baselines_df, conn, cursor)
    print(f"Selected Baseline Version: {selected_baseline_name}")
    print(f"Selected Devices: {selected_devices}")

    search_string = selected_baseline_name

    if search_string and selected_devices:
        print(f"Search String Entered: {search_string}")
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

        current_directory = os.getcwd()
        # cv_image_path = os.path.join(current_directory, '/figures/no_initial_photo_available.jpg')
        # no_data_image_path = os.path.join(current_directory, '/figures/no_data.png')

        baseline_eccheckins = get_trd_eccheckins(conn, cursor, device_id_list)
        path_list = baseline_eccheckins['server_path'].values 
        path_list = [item for item in path_list if item is not None]
        print('old ALL path list: ', path_list)
        local_all_paths = get_local_paths(path_list)
        print('new ALL path list:', local_all_paths)

        # GET THE PARENT DIRECTORIES OF EACH DEVICE
        unique_dirs = extract_unique_parent_dirs(local_all_paths)
        #unique_dirs = ['/Users/sarahpearce/Library/CloudStorage/GoogleDrive-sarah@tynt.io/Shared drives/Data/Devices/2024/07/20240703/20240703_PB_4171/']
        
        # SET EVERYTHING TO NONE FIRST 
        keyence_paths = ['None Found']
        local_warmup_paths = ['None Found']
        optics_folder_paths = ['In Progress']
        photo_folder_paths = ['None Found']
        arbin_paths = ['None Found']

        # GET THE ARBIN PATHS
        arbin_paths = []
        for path in unique_dirs:
            arbin_path = get_all_arbin_folders(path) 
            arbin_paths.append(arbin_path)
        print(unique_dirs)
        print(arbin_paths)

        # GET THE WARMUP PATHS
        baseline_warmups = get_baseline_warmups(conn, cursor, device_id_list)
        warmup_path_list = baseline_warmups['server_path'].values 
        warmup_path_list = [item for item in warmup_path_list if item is not None]
        print('old warmup path list: ', warmup_path_list)
        local_warmup_paths = get_local_paths(warmup_path_list)
        print('local warmup path list: ', local_warmup_paths) 

        # GET THE WARMUP PHOTO PATHS
        # Set all warmup values to None before looking for the data
        warmup_folder_paths = []
        warmup_ecs_corresponding_to_photos = []
        cycling_folder_paths = []
        warmup_folder_paths = []
        all_folders_list = []
        photo_folder_paths = []
        if len(unique_dirs) > 0:
            test = unique_dirs[0]

            all_folders_list = find_folders_recursively(test)
            photo_folder_paths = find_photo_folder_paths(all_folders_list)
            
            cycling_folder_paths = find_cycle_paths(photo_folder_paths)
            warmup_folder_paths = find_warmup_paths(photo_folder_paths)

            print('Warmup Photo Folders:', warmup_folder_paths)
            warmup_ecs_corresponding_to_photos = get_corresp_ec_filepaths(warmup_folder_paths)
            print('Warmup Folder EC files:', warmup_ecs_corresponding_to_photos)


        ' ############################# CALLING ALL HTML FUNCTIONS FOR DEVICE OVERVIEW TAB ################### '
        # adding route_name to baseline devices dataframe
        routes_df = get_routes(conn, cursor)
        baseline_devices = get_baseline_devices(conn, cursor, baseline_id)
        baseline_devices = pd.merge(baseline_devices, routes_df, on='route_id')

        table_html = all_devices_table(baseline_devices)
        # Bullet list of gathered data
        warmups = local_warmup_paths
        o_checks = optics_folder_paths
        p_checks = photo_folder_paths
        arbin = arbin_paths
        keyence = keyence_paths

        report_html = generate_devices_report(warmups, o_checks, p_checks, arbin, keyence)

        # final_df = get_all_raw_data(local_all_paths) # local all paths includes only things uploaded to db!!

        ' ################### GETTING ACTUAL ARBIN DATA ##################### '
        arbin_df = get_devices_arbin_checkins(conn, cursor, device_id_list)
        print('ARBIN CHECKS!!!!!!', arbin_df.columns)
        if not arbin_df.empty: # if it's not empty
            single_cycles_df = get_devices_single_cycle_arbin_checkins(conn, cursor, arbin_df)
            print('ARBIN SINGLE CHECKS!!!!!!', single_cycles_df.columns)
        else: # EMPTY!
            single_cycles_df = pd.DataFrame(data=[])

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
        if cycling_folder_paths is not None and cycling_folder_paths:
            rows = len(cycling_folder_paths)
            photo_checkin_crop_box = (450, 1200, 2400, 3300)
            image_paths = [os.path.join(cycling_folder_paths[0], fname) for fname in sorted(os.listdir(cycling_folder_paths[0])) if fname.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
            print('Accessing local gdrive cycling folder paths....', image_paths)

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
        photo_checkin_crop_box = () 
        #image_dir ='/Users/sarahpearce/Library/CloudStorage/GoogleDrive-sarah@tynt.io/Shared drives/Data/Devices/2024/07/20240703/20240703_PB_4172/precycle1/pictures'
        if warmup_folder_paths is not None and warmup_folder_paths:
            image_dir = warmup_folder_paths[0]
            photo_checkin_crop_box = (450, 1200, 2400, 3300) 
            image_paths = [os.path.join(image_dir, fname) for fname in sorted(os.listdir(image_dir)) if fname.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
            print('Accessing local gdrive warmup image paths....', image_paths)
        
            # FILEPATH FOR CORRESPONDING EC FILE TO IMAGES
            #file_path = '/Users/sarahpearce/Library/CloudStorage/GoogleDrive-sarah@tynt.io/Shared drives/Data/Devices/2024/07/20240703/20240703_PB_4172/precycle1/20240703_PB_4172_precycle1.tyntEC'
            file_path = warmup_ecs_corresponding_to_photos[0]
            photo_step_descriptions = photo_step_description(file_path)

            # Get full schedule and plot EC curve with the photos
            formatted_schedule, combined_plot = plot_data_and_print_schedule(file_path)

        def get_image_and_description(index, image_paths, descriptions):
            print(image_paths)
            if not 'no_data' in image_paths[index] and len(image_paths) > 1:
                if index < len(image_paths):
    
                    image = crop_image(image_paths[index], photo_checkin_crop_box)
                    description = descriptions[index]
                    return pn.Column(
                        pn.Row(decrement_button, slider, increment_button),
                        pn.pane.Image(image, width=600, height=400, align='start'),
                        pn.pane.Markdown(description),
                    )
            elif not 'no_data' in image_paths[index] and len(image_paths) == 1:
                image = crop_image(image_paths[index], photo_checkin_crop_box)
                description = descriptions[index]
                return pn.Column(
                        pn.pane.Image(image, width=600, height=400, align='start'),
                        pn.pane.Markdown(description),
                    )
            else:
                return pn.pane.HTML("No content available")


    ' ############################### MAKE ARBIN PLOTS ################################# '
    # Create all plots
    arbin_plots_layout = create_single_panel_plot(single_cycles_df) # handles empty dfs in function

    ' ############################### MAKE NON-CYCLE CHECKIN PLOTS ################################# '
    import matplotlib.pyplot as plt

    def create_plot_from_df(checkin_df_dict):
        plots = {}

        for name, df in checkin_df_dict.items():
            print(name, df)
            
            if name == 'df_bubbleareacheckin':
                x = df['check_in_age']
                y = df['check_in_bubblearea']
                age_units = df['check_in_age_unit'].astype(str)  # Convert to string if needed

            
                # Create the plot
                fig, ax = plt.subplots()
                unique_units = age_units.unique()
                for unit in unique_units:
                    unit_mask = age_units == unit
                    ax.scatter(x[unit_mask], y[unit_mask], label=unit)
                    
                # Add labels and legend
                ax.set_xlabel('Check In Age')
                ax.set_ylabel('Bubble Area (%)')
                ax.legend(title='Age Unit')
                ax.set_title('Device Bubble Area Check-In')

                if not df.empty:
                    # Adjust y-axis limits
                    y_min, y_max = y.min(), y.max()
                    y_range = y_max - y_min
                    print(y_range)
                    ax.set_ylim(y_min - 0.15 * y_range, y_max + 0.15 * y_range)

                unique_x = sorted(set(x))
                avg_y = [np.mean([y[i] for i in range(len(x)) if x[i] == val]) for val in unique_x]
                table_data = list(zip(unique_x, avg_y))

                if not df.empty:
                    table = plt.table(cellText=table_data, colLabels=['Cycle #', 'Average'], cellLoc='center', loc='bottom', bbox=[0.1, -0.3, 0.8, 0.2])
                
                # Save the figure
                plot_filename = f"{name}.png"
                fig.savefig(plot_filename, bbox_inches='tight')
                plt.close(fig)  # Close the figure to free memory

                plots[name] = plot_filename  # Save filename to dictionary

            elif name == 'df_devicethicknesscheckin':
                x = df['check_in_age']
                y = df['check_in_bottom_thickness_cm']
                age_units = df['check_in_age_unit'].astype(str)  # Convert to string if needed

            
                # Create the plot
                fig, ax = plt.subplots()
                unique_units = age_units.unique()
                for unit in unique_units:
                    unit_mask = age_units == unit
                    ax.scatter(x[unit_mask], y[unit_mask], label=unit)
                    
                # Add labels and legend
                ax.set_xlabel('Check In Age')
                ax.set_ylabel('Bottom Thickness (cm)')
                ax.legend(title='Age Unit')
                ax.set_title('Device Thickness Check-In')


                if not df.empty:
                    # Adjust y-axis limits
                    y_min, y_max = y.min(), y.max()
                    y_range = y_max - y_min
                    print(y_range)
                    ax.set_ylim(y_min - 0.15 * y_range, y_max + 0.15 * y_range)

                    unique_x = sorted(set(x))
                    avg_y = [np.mean([y[i] for i in range(len(x)) if x[i] == val]) for val in unique_x]
                    table_data = list(zip(unique_x, avg_y))
                    table = plt.table(cellText=table_data, colLabels=['Cycle #', 'Average'], cellLoc='center', loc='bottom', bbox=[0.1, -0.3, 0.8, 0.2])
                    
                # Save the figure
                plot_filename = f"{name}.png"
                fig.savefig(plot_filename, bbox_inches='tight')
                plt.close(fig)  # Close the figure to free memory

                plots[name] = plot_filename  # Save filename to dictionary

            elif name == 'df_hazecheckin':
                x = df['check_in_age']
                y = df['check_in_haze']
                age_units = df['check_in_age_unit'].astype(str)  # Convert to string if needed

                # Create the plot
                fig, ax = plt.subplots()
                unique_units = age_units.unique()

                for unit in unique_units:
                    unit_mask = age_units == unit
                    ax.scatter(x[unit_mask], y[unit_mask], label=unit)
                    
                # Add labels and legend
                ax.set_xlabel('Check In Age')
                ax.set_ylabel('Haze (%)')
                ax.legend(title='Age Unit')
                ax.set_title('Haze Check-In')

                                # Adjust y-axis limits

                if not df.empty:
                    # Adjust y-axis limits
                    y_min, y_max = y.min(), y.max()
                    y_range = y_max - y_min
                    ax.set_ylim(y_min - 0.15 * y_range, y_max + 0.15 * y_range)

                    unique_x = sorted(set(x))
                    avg_y = [np.mean([y[i] for i in range(len(x)) if x[i] == val]) for val in unique_x]
                    table_data = list(zip(unique_x, avg_y))
                    table = plt.table(cellText=table_data, colLabels=['Cycle #', 'Average'], cellLoc='center', loc='bottom', bbox=[0.1, -0.3, 0.8, 0.2])
                    
                # Save the figure
                plot_filename = f"{name}.png"
                fig.savefig(plot_filename, bbox_inches='tight')
                plt.close(fig)  # Close the figure to free memory


                plots[name] = plot_filename  # Save filename to dictionary
            
            elif name == 'df_weightcheckin':
                x = df['check_in_age']
                y = df['check_in_weight_g']
                age_units = df['check_in_age_unit'].astype(str)  # Convert to string if needed

                # Create the plot
                fig, ax = plt.subplots()
                unique_units = age_units.unique()
                for unit in unique_units:
                    unit_mask = age_units == unit
                    ax.scatter(x[unit_mask], y[unit_mask], label=unit)

                # Add labels and legend
                ax.set_xlabel('Check In Age')
                ax.set_ylabel('Weight (g)')
                ax.legend(title='Age Unit')
                ax.set_title('Device Weight Check-In')

                if not df.empty:
                    # Adjust y-axis limits
                    y_min, y_max = y.min(), y.max()
                    y_range = y_max - y_min
                    ax.set_ylim(y_min - 0.15 * y_range, y_max + 0.15 * y_range)

                    unique_x = sorted(set(x))
                    avg_y = [np.mean([y[i] for i in range(len(x)) if x[i] == val]) for val in unique_x]
                    table_data = list(zip(unique_x, avg_y))
                    table = plt.table(cellText=table_data, colLabels=['Cycle #', 'Average'], cellLoc='center', loc='bottom', bbox=[0.1, -0.3, 0.8, 0.2])

                # Save the figure
                plot_filename = f"{name}.png"
                fig.savefig(plot_filename, bbox_inches='tight')
                plt.close(fig)  # Close the figure to free memory

                plots[name] = plot_filename  # Save filename to dictionary

            elif name == 'df_meshwidthcheckin':
                x = df['check_in_age']
                y = df['check_in_width_um']
                age_units = df['check_in_age_unit'].astype(str)  # Convert to string if needed

                # Create the plot
                fig, ax = plt.subplots()
                unique_units = age_units.unique()
                for unit in unique_units:
                    unit_mask = age_units == unit
                    ax.scatter(x[unit_mask], y[unit_mask], label=unit)

                # Add labels and legend
                ax.set_xlabel('Check In Age')
                ax.set_ylabel('Mesh Width (um)')
                ax.legend(title='Age Unit')
                ax.set_title('Mesh Width Check-In')

                if not df.empty:
                    # Adjust y-axis limits
                    y_min, y_max = y.min(), y.max()
                    y_range = y_max - y_min
                    ax.set_ylim(y_min - 0.15 * y_range, y_max + 0.15 * y_range)

                    unique_x = sorted(set(x))
                    avg_y = [np.mean([y[i] for i in range(len(x)) if x[i] == val]) for val in unique_x]
                    table_data = list(zip(unique_x, avg_y))
                    table = plt.table(cellText=table_data, colLabels=['Cycle #', 'Average'], cellLoc='center', loc='bottom', bbox=[0.1, -0.3, 0.8, 0.2])
                    
                # Save the figure
                plot_filename = f"{name}.png"
                fig.savefig(plot_filename, bbox_inches='tight')
                plt.close(fig)  # Close the figure to free memory

                plots[name] = plot_filename  # Save filename to dictionary
            
            elif name == 'df_internalresistancecheckin':
                x = df['check_in_age']
                y = df['check_in_internal_resistance']
                age_units = df['check_in_age_unit'].astype(str)  # Convert to string if needed

                # Create the plot
                fig, ax = plt.subplots()
                unique_units = age_units.unique()
                for unit in unique_units:
                    unit_mask = age_units == unit
                    ax.scatter(x[unit_mask], y[unit_mask], label=unit)

                # Add labels and legend
                ax.set_xlabel('Check In Age')
                ax.set_ylabel('Internal Resistance (Ohm)')
                ax.legend(title='Age Unit')
                ax.set_title('Internal Resistance Check-In')

                if not df.empty:
                    # Adjust y-axis limits
                    y_min, y_max = y.min(), y.max()
                    y_range = y_max - y_min
                    ax.set_ylim(y_min - 0.15 * y_range, y_max + 0.15 * y_range)

                # Save the figure
                plot_filename = f"{name}.png"
                fig.savefig(plot_filename, bbox_inches='tight')
                plt.close(fig)  # Close the figure to free memory

                plots[name] = plot_filename  # Save filename to dictionary

            

            

        return plots

    plots = create_plot_from_df(checkin_df_dict)

    print('ALL PLOTS', plots)


    image_panes = [pn.pane.PNG(filename, width=400, height=400) for filename in plots.values()]
    noncycle_jmp_layout = pn.GridBox(*image_panes, ncols=2, sizing_mode='stretch_width')


    ' ############################### MAKE ECCHECKIN/OPTICSCHECKIN PLOTS ################################# '
    # Create all plots
    #jmp_plots_layout = create_single_panel_plot(single_cycles_df)
    ec_optics_df = get_ec_optics_joined(conn, cursor, device_id_list)
    print(ec_optics_df.columns)
    ec_optics_df = ec_optics_df.loc[:, ~ec_optics_df.columns.duplicated()]
    jmp_plots_layout = create_jmp_panel(ec_optics_df)

    y_variables = ['coulombic_efficiency', 'tint_max_current', 'tint_charge_a', 'tint_charge_b',
        'tint_charge_c', 
        'tint_max_current_time', 'delta_initial_final_percentage',
        'delta_max_min_percentage', 'final_percentage', 'initial_percentage',
        'max_percentage', 'min_percentage', 'tint_ten_time', 'tint_five_time',
        'a_initial',  'b_initial', 
        'deltaE_initial', 'deltaE_final',
        'mesh_width_checkin', 'tint_time_eighty_vlt']
    cycle_plots = {}
    ec_optics_df = ec_optics_df.fillna(np.nan)
    for y_variable in y_variables:
        x = ec_optics_df['cycle_number'].values
        x = x.astype(np.int64)
        y = ec_optics_df[y_variable].values
        print(y)
        y = y.astype(np.int64)
        print(y)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(x, y, color='black') # label='Scatter Data')
        ax.set_xticks(np.arange(0, 200, 100))
        ax.set_xlim(-50, 200)
        # Add labels
        ax.set_xlabel('Cycle #')
        ax.set_ylabel(y_variable)
        ax.set_title(y_variable + ' vs Cycle #')

        if np.issubdtype(y.dtype, np.number) and np.any(np.isfinite(y)):
            # Adjust y-axis limits
            y_min, y_max = y.min(), y.max()
            y_range = y_max - y_min
            ax.set_ylim(y_min - 0.15 * y_range, y_max + 0.15 * y_range)

        # Create lists of y-values grouped by x-values
        box_data = []
        unique_x = sorted(set(x))
        for val in unique_x:
            box_data.append([y[i] for i in range(len(x)) if x[i] == val])
        # Boxplot
        boxprops = dict(facecolor='lightgreen', color='darkgreen')  # Soft light green fill
        whiskerprops = dict(color='darkgreen')  # Dark green whiskers
        capprops = dict(color='darkgreen')  # Dark green caps

        #ax.boxplot(box_data, positions=range(1, len(unique_x) + 1), widths=0.5, patch_artist=True, 
        #    labels=[str(val) for val in unique_x], 
        #    boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops)
                # Calculate averages of y for each unique x
        #ax.legend()
        # Add table to the figure
                # Calculate averages of y for each unique x
        unique_x = sorted(set(x))
        avg_y = [np.mean([y[i] for i in range(len(x)) if x[i] == val]) for val in unique_x]
        table_data = list(zip(unique_x, avg_y))
        if not ec_optics_df.empty:
            table = plt.table(cellText=table_data, colLabels=['Cycle #', 'Average'], cellLoc='center', loc='bottom', bbox=[0.1, -0.3, 0.8, 0.2])
        
        plot_filename = f"{y_variable}.png"
        fig.savefig(plot_filename, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free memory
        cycle_plots[y_variable] = plot_filename 

    image_panes = [pn.pane.PNG(filename, width=600, height=600) for filename in cycle_plots.values()]
    cycle_jmp_layout = pn.GridBox(*image_panes, ncols=2, sizing_mode='stretch_width')
    # use ec_optics_df


    '###### DASHBOARD ######### '
    
    logo_path = os.path.join(os.getcwd(), 'figures', 'tynt_logo.png')

    # Package details of the check-in schedule
    schedule_and_plot_pane = pn.Column(
        pn.pane.Markdown('### Schedule: \n' + formatted_schedule),
        pn.pane.Markdown('### Corresponding EC file:'),
        combined_plot)
    # Create a slider widget
    slider = pn.widgets.IntSlider(name='Check-in Photo #', start=0, end=len(image_paths) - 1, step=1)
    # Bind the slider to the function with additional parameters
    image_pane = pn.bind(get_image_and_description, slider, image_paths, photo_step_descriptions)
    # Define button callbacks to adjust slider value
    def increment_slider(event):
        slider.value = min(slider.end, slider.value + 1)

    def decrement_slider(event):
        slider.value = max(slider.start, slider.value - 1)

    # Create functional arrow buttons
    increment_button = pn.widgets.Button(name="▶", button_type="primary")
    decrement_button = pn.widgets.Button(name="◀", button_type="primary")

    # Attach callbacks to buttons
    increment_button.on_click(increment_slider)
    decrement_button.on_click(decrement_slider)


    # Define the main content area
    main_content = pn.Column(pn.Column('## All Devices in Baseline Run:', pn.pane.HTML(table_html),
        pn.Column('## Warmup Data', pn.Row(image_pane, schedule_and_plot_pane),),
        pn.Column('## Cycling Data',))
    )

    section1 = pn.Column('## All Devices in Baseline Run:', pn.pane.HTML(table_html), pn.pane.HTML(report_html))
    section2 = pn.Column('## Warmup Data', pn.Row(image_pane, schedule_and_plot_pane),)
    section3 = pn.Column('## Cycling Data',)
    section4 = pn.Column('## Checkin Summary Values from Database', 
                         pn.Row(pn.Column('## Interactive Plots:'), pn.Column(jmp_plots_layout)),
                         pn.Row(pn.Row('## Static Plots (Non-cycle Checkins):'), 
                                pn.Row(noncycle_jmp_layout)),
                        pn.Row(pn.Row('## Static Plots (Cycing Checkins):'), 
                                                    pn.Row(cycle_jmp_layout)))
    section5 = pn.Column('## Arbin Summary Values from Database', arbin_plots_layout)
    section6 = pn.Column('## Keyence Images',)
    section7 = pn.Column('## Durability Predictions (keras/tensorflow modeling)',)

    main_content = pn.Tabs(
        ('Devices', section1),
        ('Warmup Data', section2),
        ('Cycling Data', section3),
        ('JMP Summary Plots', section4),
        ('Arbin Summary Plots', section5),
        ('Keyence Images', section6),
        ('Predictive Modeling', section7),
    )
    
        # Define custom CSS for 3D effect

    # Create buttons in the sidebar to navigate to each section
    button1 = pn.widgets.Button(name='Go to Device Details', button_type='primary')
    button2 = pn.widgets.Button(name='Go to Warmup Data', button_type='primary')
    button3 = pn.widgets.Button(name='Go to Raw Cycling Data', button_type='primary')
    button4 = pn.widgets.Button(name='Go to Summarized Checkin Cycling Data', button_type='primary')
    button5 = pn.widgets.Button(name='Go to Summarized Arbin Cycling Data', button_type='primary')
    button6 = pn.widgets.Button(name='Go to Durability Predictions', button_type='primary')
    button7 = pn.widgets.Button(name='Go to Keyence Images', button_type='primary')
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
    # Attach callbacks to buttons
    button1.on_click(go_to_section1)
    button2.on_click(go_to_section2)
    button3.on_click(go_to_section3)
    button4.on_click(go_to_section4)
    button5.on_click(go_to_section5)
    button6.on_click(go_to_section6)
    button7.on_click(go_to_section7)

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

