import panel as pn
import pandas as pd
import psycopg2
import hvplot.pandas
import holoviews as hv
from holoviews import opts
import os
import glob
import numpy as np
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QComboBox, QPushButton, QLabel
import sys
import re
import holoviews as hv
import pandas as pd
from holoviews import opts
import numpy as np
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QComboBox, QPushButton, QLabel
import sys
from pathlib import Path
import re
import panel as pn
from PIL import Image
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QComboBox, QPushButton, QLabel
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QComboBox, QPushButton, QLabel, QCheckBox, QScrollArea, QFormLayout


def get_user_input():
    # Create a QApplication instance
    app = QApplication([])

    # Create a QWidget as the main window
    window = QWidget()

    # Use QInputDialog to get user input
    user_input, ok = QInputDialog.getText(window, 'Input', 'Please enter a string:')

    # Check if the user pressed OK
    if ok:
        print(f"You entered: {user_input}")

    return user_input


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

def search_baseline_name(df, search_string):
    # Ensure 'baseline_version' is treated as string
    df['baseline_version'] = df['baseline_version'].astype(str)
    
    # Filter rows where 'baseline_version' exactly matches the search string
    filtered_df = df[df['baseline_version'] == search_string]
    
    # Extract corresponding 'id' values
    ids = filtered_df['id'].tolist()
    
    return ids


def get_routes(conn, cursor):
    '''baseline DF has ids for route ids but not the routes themselves, so we need to map these'''
    if conn and cursor:
        # Update SQL query to filter by trd_id
        sql_query = "SELECT id, route_name FROM tyntdatabase_route;"
        # Execute the query with trd_id parameter
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        route_data = [{'route_id': row[0], 'route_name': row[1]} for row in rows]
        routes_df = pd.DataFrame(route_data)

    return routes_df


def get_baseline_devices(conn, cursor, baseline_id):
    if conn and cursor:
        # Update SQL query to filter by trd_id
        sql_query = '''
            SELECT * 
            FROM tyntdatabase_device
            WHERE baseline_version_id = %s
            LIMIT ALL 
            OFFSET 0;
        '''
        # Execute the query with trd_id parameter
        cursor.execute(sql_query, (baseline_id,))
        devices = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        devices = pd.DataFrame(devices, columns=column_names)

    return devices

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
        'E:\\',
        'T:\\'                                              # Root of E: drive, adjust if needed
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

    print('system', system)

    target_directory = 'Shared drives'
    sub_directory = 'Data'

    for base_path in base_paths:
        print('base path search...', base_path)
        path = find_directory(base_path, target_directory)
        print('finding directory....', path)
        if path:
            # Check within the found directory for the sub-directory 'Data'
            data_path = find_directory(path, sub_directory)
            if data_path:
                return data_path

    return None

from pathlib import Path

def get_all_arbin_folders(directory):
    # Extract the last folder name from the directory path
    last_folder_name = os.path.basename(os.path.normpath(directory))
    
    # List to store matching subfolders
    matching_subfolders = []
    
    # Get the parent directory
    parent_directory = os.path.dirname(directory)
    
    # Iterate over all items in the parent directory
    for item in os.listdir(parent_directory):
        item_path = os.path.join(parent_directory, item)
        # Check if it's a directory and if the last folder's name is a substring of the item name
        if os.path.isdir(item_path) and last_folder_name in item:
            item_path = Path(str(item_path).replace(' (7:24:24)', ''))
            matching_subfolders.append(item_path)
    
    return matching_subfolders

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
    result = clean_path(result)
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
        substring = 'google-drive-data'
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
        final_path = clean_path(final_path)
        local_paths.append(final_path)

    return local_paths

def clean_path(path):
    path = Path(str(path).replace(' (7:24:24)', ''))
    return path

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

'################## WARMUP BASELINE FUNCTIONS ############################### '

def get_baselines(conn, cursor):
    if conn and cursor:
        sql_query = '''
            SELECT * 
            FROM tyntdatabase_baselineversion
            LIMIT ALL 
            OFFSET 0;
        '''
        cursor.execute(sql_query)
        baselines = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        baselines = pd.DataFrame(baselines, columns=column_names)

    return baselines

def create_dynamic_baseline_dialog(df, conn, cursor):
    # Create the application and the main window
    app = QApplication([])
    window = QWidget()
    window.setWindowTitle("Select Baseline Version and Devices to Plot")

    # Create a layout for the main window
    main_layout = QVBoxLayout(window)

    # Label and first dropdown
    label = QLabel("Select a Baseline Version:")
    main_layout.addWidget(label)

    first_dropdown = QComboBox()
    first_dropdown.addItems(reversed(df['baseline_version'].values)) 
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
        matching_ids = search_baseline_name(df, category)
        baseline_id = matching_ids[0]
        baseline_devices = get_baseline_devices(conn, cursor, baseline_id)
        device_list = baseline_devices['device_name'].values  # Assuming trd_devices has 'device_name' column

        for device in device_list:
            checkbox = QCheckBox(device)
            checkbox.setChecked(True)
            checkbox_layout.addWidget(checkbox)

    # Connect the first dropdown's change event to the update function
    first_dropdown.currentTextChanged.connect(update_checkboxes)

    # Initialize the checkboxes for the first time
    update_checkboxes()

    # Variables to hold the selection
    selected_baseline_name = None
    selected_devices = []

    # Function to capture the selections
    def capture_selections():
        nonlocal selected_baseline_name, selected_devices
        selected_baseline_name = first_dropdown.currentText()
        selected_devices = [checkbox.text() for checkbox in checkbox_widget.findChildren(QCheckBox) if checkbox.isChecked()]
        print(f"Selected Baseline Name: {selected_baseline_name}")
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
    return selected_baseline_name, selected_devices

from PIL import Image
def create_gif(image_paths, output_path, duration=500):
    """
    Create a GIF from a list of image paths and save it to a file.

    :param image_paths: List of file paths to the images.
    :param output_path: Path to save the resulting GIF.
    :param duration: Duration for each frame in milliseconds (default is 500ms).
    """
    # Load images
    images = [Image.open(image_path) for image_path in image_paths]

    # Save as GIF
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )

    return output_path

import io
import bokeh.plotting as bkp
import bokeh.models as bkm

import pandas as pd
import hvplot.pandas
import bokeh.plotting as bkp
import bokeh.models as bkm

def plot_data_and_print_schedule(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Find the index of the "# Schedule" line
    schedule_start_index = next(i for i, line in enumerate(lines) if line.strip() == "# Schedule")
    
    # Extract the lines from "# Schedule" to the end of the file
    schedule_lines = lines[schedule_start_index+1:]
    
    # Find the index where the actual data starts (after schedule lines)
    data_start_index = next((i for i, line in enumerate(schedule_lines) if not line.startswith('#')), len(schedule_lines))
    
    # Extract and process the schedule section
    schedule_lines = schedule_lines[:data_start_index]
    
    # Filter out lines containing "Capture Photo"
    filtered_lines = [line.strip() for line in schedule_lines if "Capture Photo" not in line]
    
    # Format the output
    formatted_lines = [line.lstrip('#').strip() for line in filtered_lines]
    formatted_schedule = '\n'.join(formatted_lines).strip()

    df = pd.read_csv(file_path, sep='\t', comment='#', index_col=0)
    time = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S.%f')
    startTime = time[0]
    time -= startTime
    time = time.dt.total_seconds()
    current = df['Current (A)']
    current = current * 1000  # convert from A to mA
    df['Current (mA)'] = current
    df['Time'] = time
    
    # Coulombic Efficiency Calculation
    charge = df['Charge (C)']
    qIn = charge.min()
    qFinal = charge.iloc[-1]
    qOut = qFinal-qIn
    coulombic_efficiency = abs(qOut/qIn)*100
    
    # Define the plots
    voltage_plot = df.hvplot.line(x='Time', y='Voltage (V)', color='blue', label='Voltage (V)', width=600, height=300)
    current_plot = df.hvplot.line(x='Time', y='Current (mA)', color='red', label='Current (mA)', width=600, height=300)
    charge_plot = df.hvplot.line(x='Time', y='Charge (C)', color='green', label='Charge (C)', width=600, height=300)
    
    # Create Bokeh figure
    p = bkp.figure(width=600, height=300, title=None)

    # Define ranges for each y-axis
    voltage_range = bkm.Range1d(start=df['Voltage (V)'].min() - 0.1, end=df['Voltage (V)'].max() + 0.1)
    current_range = bkm.Range1d(start=df['Current (mA)'].min() - 10, end=df['Current (mA)'].max() + 10)
    charge_range = bkm.Range1d(start=df['Charge (C)'].min() - 0.05, end=df['Charge (C)'].max() + 0.05)

    # Plot Voltage
    p.line(df['Time'], df['Voltage (V)'], color='blue', legend_label='Voltage (V)')
    p.yaxis.axis_label = 'Voltage (V)'
    p.y_range = voltage_range  # Set range for the Voltage axis

    # Add additional y-axes for Current and Charge
    p.extra_y_ranges = {
        'current': current_range,
        'charge': charge_range
    }

    p.add_layout(bkm.LinearAxis(y_range_name='current', axis_label='Current (mA)'), 'right')
    p.add_layout(bkm.LinearAxis(y_range_name='charge', axis_label='Charge (C)'), 'right')

    # Plot Current
    p.line(df['Time'], df['Current (mA)'], color='red', legend_label='Current (mA)', y_range_name='current')

    # Plot Charge
    p.line(df['Time'], df['Charge (C)'], color='green', legend_label='Charge (C)', y_range_name='charge')

    # Add grid and legend
    p.add_tools(bkm.HoverTool())
    p.legend.location = 'bottom_right'
    p.grid.grid_line_alpha = 0.3

    # Return the schedule, plot, and calculated Coulombic Efficiency
    return formatted_schedule, p, f"Coulombic Efficiency = {coulombic_efficiency:.2f}%"




def photo_step_description(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    step_info = ""
    results = []

    for line in lines:
        # Check if the line contains a step
        step_match = re.match(r'^# Step (\d+): (.+)', line)
        if step_match:
            step_number = step_match.group(1)
            step_description = step_match.group(2)
            # Keep only the initial part of the description before the first "|"
            step_description = step_description.split(' | ')[0]
            step_info = f"Step {step_number}: {step_description}"
        # Check if the line contains "Action: Capture Photo"
        elif "Action: Capture Photo" in line:
            # Extract the condition
            condition_match = re.search(r'Cutoff \d+: (.+?) \| Action: Capture Photo', line)
            if condition_match:
                condition = condition_match.group(1)
                results.append(f"{step_info}, {condition}")

    return results

# Example usage
#image_paths = ['image1.png', 'image2.png', 'image3.png']  # Replace with your image paths
#output_path = 'output.gif'  # Replace with your desired output file path
#create_gif(image_paths, output_path, duration=500)

def extract_headers(file_path):
    """
    Extract lines that start with '#' from a text file.

    :param file_path: Path to the text file.
    :return: List of header lines.
    """
    headers = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # Strip leading/trailing whitespace and check if the line starts with '#'
                if line.strip().startswith('#'):
                    headers.append(line.strip())
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    return headers


def get_warmup_schedule(filePath):
    result = ''
    with open(filePath, 'r') as file:
        content = file.read()
        lines = content.splitlines()
        # List to hold extracted steps
        steps = []
        for line in lines:
            if line.startswith('# Step'):
                # Save previous step text if it exists
                steps.append(line)
    step_string = ''
    for step in steps:
        step_string = step_string + step
        step_string = step_string + '\n'
    return step_string

def warmup_paths_only(local_paths):
    """
    Already converted server paths stored in database to local paths.
    Now taking a subset of those that correspond to warmups for the pre-cycle seciton of the Dashboard.
    """
    warmup_paths_only = []
    for path in local_paths:
        if 'precycle' in path:
            warmup_paths_only.append(path)

    return warmup_paths_only

def get_gif_path(folder):
    files = os.listdir(folder)
    gif_path = ''
    for f in files:
        if f.endswith('.gif'):
            if 'cycle' in f:
                gif_path = os.path.join(folder, f)
    return gif_path

def get_baseline_warmups(conn, cursor, device_id_list):
    if conn and cursor:
        device_id_list = [int(id) for id in device_id_list]
        print(device_id_list)
        # Convert the list of IDs to a format suitable for SQL IN clause
        format_strings = ','.join(['%s'] * len(device_id_list))
        print(format_strings)
        sql_query = f'''
            SELECT * 
            FROM tyntdatabase_warmupeccheckin
            WHERE device_id IN ({format_strings})
            LIMIT ALL 
            OFFSET 0;
        '''
        # Execute the query with the list of IDs as parameters
        cursor.execute(sql_query, device_id_list)
        baseline_warmups = cursor.fetchall()
        print(baseline_warmups)
        column_names = [desc[0] for desc in cursor.description]
        baseline_warmups_df = pd.DataFrame(baseline_warmups, columns=column_names)

    return baseline_warmups_df

def get_devices_arbin_checkins(conn, cursor, device_id_list):
    if conn and cursor:
        device_id_list = [int(id) for id in device_id_list]
        print(device_id_list)
        # Convert the list of IDs to a format suitable for SQL IN clause
        format_strings = ','.join(['%s'] * len(device_id_list))
        print(format_strings)
        sql_query = f'''
            SELECT * 
            FROM tyntdatabase_arbincheckin
            WHERE device_id IN ({format_strings})
            LIMIT ALL 
            OFFSET 0;
        '''
        # Execute the query with the list of IDs as parameters
        cursor.execute(sql_query, device_id_list)
        arbin_checkins = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        arbin_checkins_df = pd.DataFrame(arbin_checkins, columns=column_names)

    return arbin_checkins_df

def get_devices_single_cycle_arbin_checkins(conn, cursor, arbin_checkins_df):
    arbin_id_list = arbin_checkins_df['id'].tolist()
    print('ARBIN IDS EXTRACTED:', arbin_id_list)
    if conn and cursor:
        arbin_id_list = [int(id) for id in arbin_id_list]
        # Convert the list of IDs to a format suitable for SQL IN clause
        format_strings = ','.join(['%s'] * len(arbin_id_list))
        print(format_strings)
        sql_query = f'''
            SELECT * 
            FROM tyntdatabase_singlecyclearbincheckin
            WHERE arbincheckin_id IN ({format_strings})
            LIMIT ALL 
            OFFSET 0;
        '''
        # Execute the query with the list of IDs as parameters
        cursor.execute(sql_query, arbin_id_list)
        single_cycle_arbin_checkins = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        single_cycle_arbin_checkins_df = pd.DataFrame(single_cycle_arbin_checkins, columns=column_names)

    return single_cycle_arbin_checkins_df

import pandas as pd

def get_ec_optics_joined(conn, cursor, device_id_list):
    if conn and cursor:
        device_id_list = [int(id) for id in device_id_list]
        print(device_id_list)
        # Convert the list of IDs to a format suitable for SQL IN clause
        format_strings = ','.join(['%s'] * len(device_id_list))
        print(format_strings)
        
        sql_query = f'''
            SELECT eccheckin.*, opticscheckin.*
            FROM tyntdatabase_eccheckin AS eccheckin
            JOIN tyntdatabase_opticscheckin AS opticscheckin
            ON eccheckin.device_id = opticscheckin.device_id
            AND eccheckin.cycle_number = opticscheckin.cycle_number
            WHERE eccheckin.device_id IN ({format_strings})
            LIMIT ALL 
            OFFSET 0;
        '''
        
        # Execute the query with the list of IDs as parameters
        cursor.execute(sql_query, device_id_list)
        trd_eccheckins = cursor.fetchall()
        print(trd_eccheckins)
        
        column_names = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(trd_eccheckins, columns=column_names)

    return df

def get_haze_weight_meshwidth_devicewidth_bubbles_ir_joined(conn, cursor, device_id_list):
    table_list = ['BubbleAreaCheckIn', 'DeviceThicknessCheckIn', 'HazeCheckIn', 
                  'WeightCheckIn', 'MeshWidthCheckIn', 'InternalResistanceCheckIn']
    # Dictionary to hold DataFrames
    dataframes = {}
    # InternalResistanceCheckIn
    # device, check_in_age, check_in_age_unit
    if conn and cursor:
        device_id_list = [int(id) for id in device_id_list]
        format_strings = ','.join(['%s'] * len(device_id_list))
        for table in table_list:
            sql_query = f'''
                SELECT * 
                FROM tyntdatabase_{table.lower()}
                WHERE device_id IN ({format_strings})
                LIMIT ALL 
                OFFSET 0;
            '''
            # Execute the query with the list of IDs as parameters
            cursor.execute(sql_query, device_id_list)
            checkins = cursor.fetchall()
            
            column_names = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(checkins, columns=column_names)
            dataframes[f'df_{table.lower()}'] = df
    return dataframes

' ############ JMP PLOT GENERATION #################### '

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import panel as pn
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label

import matplotlib.pyplot as plt
import numpy as np
import panel as pn

def plot_boxplot_with_scatter(df, y_variables):
    cycle_plots = {}
    
    # Replace None with np.nan
    df = df.replace({None: np.nan})
    
    # Define marker styles for different routes
    marker_styles = {
        'Ambient': 'o',  # Circle
        'Demo': 's',  # Square
        'Oven': '^',  # Triangle
        'Samples': 'D',  # Diamond
        'Weatherometer': 'x',  # X
    }
    
    for y_variable in y_variables:
        fig, ax = plt.subplots(figsize=(12, 6))  # Increase figure width for space for legend
        
        # Extract x, y, and route values
        x = df['cycle_number'].values
        y = df[y_variable].values
        route_names = df['route_name'].astype(str).values
        
        # Check if x or y is empty
        if len(x) == 0 or len(y) == 0 or np.all(np.isnan(y)):
            # If empty, create a placeholder plot or skip
            print(f"No data available for {y_variable}. Skipping plot.")
            plt.close(fig)  # Close the figure to free memory
            cycle_plots[y_variable] = None
        else:
            # Group y values by x values
            unique_x = sorted(set(x))
            box_data = [y[x == val] for val in unique_x]
            
            # Create boxplot with light grey fill
            box = ax.boxplot(box_data, positions=np.arange(1, len(unique_x) + 1), widths=0.6, patch_artist=True, showmeans=True)
            for patch in box['boxes']:
                patch.set(facecolor='lightgrey')
        
            # Overlay scatter plot with different marker shapes based on route_name
            markers = {}
            for xi, yi, route in zip(x, y, route_names):
                pos = np.where(np.array(unique_x) == xi)[0][0] + 1
                marker = marker_styles.get(route, 'o')  # Default to circle if route not found
                if route not in markers:
                    markers[route] = ax.scatter([], [], color='black', marker=marker, label=route)  # Initialize legend entry
                ax.scatter(pos, yi, color='black', marker=marker)
            
            # Add legend to the right of the plot
            ax.legend(handles=[markers[route] for route in markers], loc='center left', bbox_to_anchor=(1, 0.5), title='Route Name')
            
            # Calculate and display the mean for each group
            for i, pos in enumerate(np.arange(1, len(unique_x) + 1)):
                mean = np.nanmean(box_data[i])
                ax.text(pos, mean, f'{mean:.3f}', ha='left', va='center', fontsize=10, color='black', fontweight='bold')

            # Customize the plot
            ax.set_xticks(np.arange(1, len(unique_x) + 1))
            ax.set_xticklabels([str(val) for val in unique_x])
            ax.set_xlabel('Cycle #')
            ax.set_ylabel(y_variable)
            ax.set_title(f'{y_variable} vs Cycle #')
            
            # Save plot
            plot_filename = f"{y_variable}.png"
            fig.savefig(plot_filename, bbox_inches='tight')
            plt.close(fig)  # Close the figure to free memory
            cycle_plots[y_variable] = plot_filename

            print(f"Length of box_data: {len(box_data)}")
            print(f"Length of positions (unique_x): {len(unique_x)}")
    
    return cycle_plots






def make_jmp_box(data, x, y, color_col, title):
    # Convert color column to string type for grouping
    data[color_col] = data[color_col].astype(str)
    
    # Create box plots with dots for the color column
    grouped = data.groupby(color_col)
    box_plots = []
    scatter_plots = []
    
    for name, group in grouped:
        box_plot = hv.BoxWhisker(group, kdims=x, vdims=y).opts(
            opts.BoxWhisker(width=400, height=200, tools=['hover'], box_color='blue')
        )
        dot_plot = hv.Scatter(group, kdims=x, vdims=y).opts(
            opts.Scatter(size=5, color='black')
        )
        box_plots.append(box_plot)
        scatter_plots.append(dot_plot)
    
    plot = hv.Overlay(box_plots + scatter_plots).opts(
        opts.Overlay(legend_position='right', title=title)
    )
    
    return plot

import matplotlib.pyplot as plt

def create_jmp_panel(ec_optics_df):
    # Define the list of y-values
    y_values = [
        'coulombic_efficiency', 'bleach_final_current', 'bleach_max_current',
        'tint_final_current', 'tint_max_current', 'tint_charge_a', 'tint_charge_b',
        'tint_charge_c', 'tint_charge_d', 'charge_in', 'charge_out', 
        'tint_max_current_time', 'bleach_ten_time', 'bleach_five_time', 
        'bleach_one_time', 'bleach_point_one_time', 'delta_initial_final_percentage',
        'delta_max_min_percentage', 'final_percentage', 'initial_percentage',
        'max_percentage', 'min_percentage', 'tint_ten_time', 'tint_five_time',
        'tint_one_time', 'tint_point_one_time', 'tint_ten_a', 'tint_five_a',
        'tint_one_a', 'tint_point_one_a', 'a_final', 'a_initial', 'a_max',
        'a_min', 'tint_ten_b', 'tint_five_b', 'tint_one_b', 'tint_point_one_b',
        'b_final', 'b_initial', 'b_max', 'b_min', 'tint_ten_cri', 'tint_five_cri',
        'tint_one_cri', 'tint_point_one_cri', 'cri_final', 'cri_initial',
        'cri_max', 'cri_min', 'tint_ten_l', 'tint_five_l', 'tint_one_l',
        'tint_point_one_l', 'l_final', 'l_initial', 'l_max', 'l_min',
        'deltaE_initial', 'local_path', 'server_path', 'deltaE_final',
        'tint_five_deltaE', 'deltaE_min', 'tint_one_deltaE',
        'tint_point_one_deltaE', 'tint_ten_deltaE', 'expected_VLT',
        'mesh_width_checkin', 'tint_time_eighty_vlt'
    ]

    # Create a select widget for y-values
    y_select = pn.widgets.Select(name='Select Y Variable', options=y_values, value=y_values[0])

    # Define a callback function to update the plot based on the selected y_value
    @pn.depends(y_value=y_select)
    def update_plot(y_value):
        if y_value is None:
            return "Please select a y-value."
        
        if 'cycle_number' not in ec_optics_df.columns:
            return "Error: 'cycle_number' column is missing from the DataFrame."

        # Call the make_row_box function with the color column
        plot = make_jmp_box(
            ec_optics_df, 'cycle_number', y_value, 'device_id', y_value
        )
        
        # Return the plot
        return plot

    # Create and return the Panel layout with the select widget and the plot
    layout = pn.Column(y_select, update_plot)
    return layout

import pandas as pd
import panel as pn

import panel as pn
import pandas as pd

def create_interactive_jmp_panel(ec_optics_df):
    # First check that the entire dataframe is not empty 
    if ec_optics_df.empty:
        return pn.pane.Markdown("# No Available Data for Interactive Plotting")

    # Define the list of y-values
    y_values = [
        'coulombic_efficiency', 'bleach_final_current', 'bleach_max_current',
        'tint_final_current', 'tint_max_current', 'tint_charge_a', 'tint_charge_b',
        'tint_charge_c', 'tint_charge_d', 'charge_in', 'charge_out', 
        'tint_max_current_time', 'bleach_ten_time', 'bleach_five_time', 
        'bleach_one_time', 'bleach_point_one_time', 'delta_initial_final_percentage',
        'delta_max_min_percentage', 'final_percentage', 'initial_percentage',
        'max_percentage', 'min_percentage', 'tint_ten_time', 'tint_five_time',
        'tint_one_time', 'tint_point_one_time', 'tint_ten_a', 'tint_five_a',
        'tint_one_a', 'tint_point_one_a', 'a_final', 'a_initial', 'a_max',
        'a_min', 'tint_ten_b', 'tint_five_b', 'tint_one_b', 'tint_point_one_b',
        'b_final', 'b_initial', 'b_max', 'b_min', 'tint_ten_cri', 'tint_five_cri',
        'tint_one_cri', 'tint_point_one_cri', 'cri_final', 'cri_initial',
        'cri_max', 'cri_min', 'tint_ten_l', 'tint_five_l', 'tint_one_l',
        'tint_point_one_l', 'l_final', 'l_initial', 'l_max', 'l_min',
        'deltaE_initial', 'local_path', 'server_path', 'deltaE_final',
        'tint_five_deltaE', 'deltaE_min', 'tint_one_deltaE',
        'tint_point_one_deltaE', 'tint_ten_deltaE', 'expected_VLT',
        'mesh_width_checkin', 'tint_time_eighty_vlt'
    ]

    # Create a select widget for route selection
    route_values = ['Plot All'] + ec_optics_df['route_name'].unique().tolist()
    route_select = pn.widgets.Select(name='Select Route', options=route_values, value=route_values[0])

    # To avoid ValueErrors, filter to only select from and generate plots for y variables that are not empty in the df
    y_values = [y for y in y_values if y in ec_optics_df.columns and ec_optics_df[y].notna().sum() > 0]

    # Create a dynamic select widget for y-values based on the filtered DataFrame
    y_select = pn.widgets.Select(name='Select Y Variable', options=y_values, value=y_values[0])

    # Function to filter DataFrame based on the selected route
    def filter_dataframe(route_name):
        if route_name == 'Plot All':
            return ec_optics_df  # Use the whole DataFrame
        else:
            return ec_optics_df[ec_optics_df['route_name'] == route_name]  # Filter by selected route_name

    # Update the plot based on selected route and y-value
    @pn.depends(route_select.param.value, y_select.param.value)
    def update_plot(route, y_value):
        filtered_df = filter_dataframe(route)
        
        if y_value is None:
            return "Please select a y-value and ensure data is available."

        if 'cycle_number' not in filtered_df.columns:
            return "Error: 'cycle_number' column is missing from the DataFrame."

        # Call the make_jmp_box function with the filtered DataFrame
        plot = make_jmp_box(
            filtered_df, 'cycle_number', y_value, 'device_id', y_value
        )
        
        return plot

    # Layout for the Panel dashboard
    panel_layout = pn.Column(route_select, y_select, pn.panel(update_plot))
    return panel_layout



# Dictionary of y-values with corresponding slider ranges
y_value_ranges = {
    'coulombic_efficiency': (-100, 100), 'bleach_final_current': (-100, 100),
    'bleach_max_current': (-100, 100), 'tint_final_current': (-100, 100),
    'tint_max_current': (-100, 100), 'tint_charge_a': (-100, 100),
    'tint_charge_b': (-100, 100), 'tint_charge_c': (-100, 100),
    'tint_charge_d': (-100, 100), 'charge_in': (-100, 100),
    'charge_out': (-100, 100), 'tint_max_current_time': (-100, 100),
    'bleach_ten_time': (-100, 100), 'bleach_five_time': (-100, 100),
    'bleach_one_time': (-100, 100), 'bleach_point_one_time': (-100, 100),
    'delta_initial_final_percentage': (-100, 100), 'delta_max_min_percentage': (-100, 100),
    'final_percentage': (-100, 100), 'initial_percentage': (-100, 100),
    'max_percentage': (-100, 100), 'min_percentage': (-100, 100),
    'tint_ten_time': (-100, 100), 'tint_five_time': (-100, 100),
    'tint_one_time': (-100, 100), 'tint_point_one_time': (-100, 100),
    'tint_ten_a': (-100, 100), 'tint_five_a': (-100, 100),
    'tint_one_a': (-100, 100), 'tint_point_one_a': (-100, 100),
    'a_final': (-100, 100), 'a_initial': (-100, 100), 'a_max': (-100, 100),
    'a_min': (-100, 100), 'tint_ten_b': (-100, 100), 'tint_five_b': (-100, 100),
    'tint_one_b': (-100, 100), 'tint_point_one_b': (-100, 100),
    'b_final': (-100, 100), 'b_initial': (-100, 100), 'b_max': (-100, 100),
    'b_min': (-100, 100), 'tint_ten_cri': (-100, 100), 'tint_five_cri': (-100, 100),
    'tint_one_cri': (-100, 100), 'tint_point_one_cri': (-100, 100),
    'cri_final': (-100, 100), 'cri_initial': (-100, 100), 'cri_max': (-100, 100),
    'cri_min': (-100, 100), 'tint_ten_l': (-100, 100), 'tint_five_l': (-100, 100),
    'tint_one_l': (-100, 100), 'tint_point_one_l': (-100, 100), 'l_final': (-100, 100),
    'l_initial': (-100, 100), 'l_max': (-100, 100), 'l_min': (-100, 100),
    'deltaE_initial': (-100, 100), 'local_path': (-100, 100), 'server_path': (-100, 100),
    'deltaE_final': (-100, 100), 'tint_five_deltaE': (-100, 100), 'deltaE_min': (-100, 100),
    'tint_one_deltaE': (-100, 100), 'tint_point_one_deltaE': (-100, 100),
    'tint_ten_deltaE': (-100, 100), 'expected_VLT': (-100, 100),
    'mesh_width_checkin': (-100, 100), 'tint_time_eighty_vlt': (-100, 100)
}

import panel as pn
import pandas as pd

def create_static_cycling_jmp_panel(df):
    # Updated code that needs to be put into a separate function 9/25/24
    # This is the most up to date version
    y_variables = ['coulombic_efficiency', 'tint_max_current', 'tint_charge_a', 'tint_charge_b',
                'tint_charge_c', 'tint_max_current_time', 'delta_initial_final_percentage',
                'delta_max_min_percentage', 'final_percentage', 'initial_percentage',
                'max_percentage', 'min_percentage', 'tint_ten_time', 'tint_five_time',
                'a_initial', 'b_initial', 'deltaE_initial', 'deltaE_final',
                'mesh_width_checkin', 'tint_time_eighty_vlt',
                'tint_ten_b', 'tint_ten_a']
    # Remove y variables without data in the dataframe
    y_variables = [y for y in y_variables if y in df.columns and df[y].notna().sum() > 0]
    if not df.empty:
        cycle_plots = plot_boxplot_with_scatter(df, y_variables)
        # Track variables with None values
        variables_with_none = [var for var, filename in cycle_plots.items() if filename is None]
        # Create image panes, skipping None values
        image_panes = [pn.pane.PNG(filename, width=600, height=400) for filename in cycle_plots.values() if filename is not None]
        # Display a report if there are variables with None
        if variables_with_none:
            print('NO DATA WARNING: Some boxplots have no JMP data')
            none_report = f"Variables with no data: {', '.join(variables_with_none)}"
            image_panes.append(pn.pane.Markdown(f"### {none_report}"))
    else: 
        print('NO DATA WARNING: No JMP Data for Static Plots')
        image_panes = [pn.Column('### No JMP Data to Display')]

    # Create a Panel layout to display plots
    cycle_jmp_layout = pn.GridBox(*image_panes, ncols=2, sizing_mode='stretch_width')

    return cycle_jmp_layout

# CURRENT STATIC NON-CYCLING JMP PLOT FUNCTION
def create_static_noncycling_plot_dictionary_from_df(checkin_df_dict, baseline_devices):
    plots = {}
    marker_styles = {
    'Ambient': 'o',  # Circle
    'Demo': 's',  # Square
    'Oven': '^',  # Triangle
    'Samples': 'D',  # Diamond
    'Weatherometer': 'x',} # X

    for name, df in checkin_df_dict.items():
        #print(name, df)

        #print(baseline_devices.columns)
        #print(df.columns)
        baseline_devices = baseline_devices.rename(columns={'id': 'device_id'})
        #print(baseline_devices.columns)
        #print(df.columns)
        df_with_route = pd.merge(baseline_devices, df, on='device_id')
        #print('route df', df_with_route)
        df = df_with_route
        
        if name == 'df_bubbleareacheckin':
            x = df['check_in_age']
            y = df['check_in_bubblearea']
            age_units = df['check_in_age_unit'].astype(str)  # Convert to string if needed
            route_names = df['route_name']  # Assuming this is the column with route names

            # Create the plot
            fig, ax = plt.subplots()

            # Loop over unique age units
            unique_units = age_units.unique()
            for unit in unique_units:
                unit_mask = age_units == unit
                for route in route_names[unit_mask].unique():
                    route_mask = route_names == route
                    combined_mask = unit_mask & route_mask
                    ax.scatter(
                        x[combined_mask], 
                        y[combined_mask], 
                        label=f"{unit} - {route}",
                        marker=marker_styles.get(route, 'o')  # Default to 'o' if route not in dictionary
                    )
            
            # Add labels and legend
            ax.set_xlabel('Check In Age')
            ax.set_ylabel('Bubble Area (%)')
            ax.legend(title='Age Unit and Route Name')
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
            route_names = df['route_name'] 

            # Create the plot
            fig, ax = plt.subplots()
                # Loop over unique age units
            unique_units = age_units.unique()
            for unit in unique_units:
                unit_mask = age_units == unit
                for route in route_names[unit_mask].unique():
                    route_mask = route_names == route
                    combined_mask = unit_mask & route_mask
                    ax.scatter(
                        x[combined_mask], 
                        y[combined_mask], 
                        label=f"{unit} - {route}",
                        marker=marker_styles.get(route, 'o')  # Default to 'o' if route not in dictionary
                    )

            # Add labels and legend
            ax.set_xlabel('Check In Age')
            ax.set_ylabel('Bottom Thickness (cm)')
            ax.legend(title='Age Unit and Route Name')
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
            route_names = df['route_name']  # Assuming this is the column with route names

            # Create the plot
            fig, ax = plt.subplots()

            unique_units = age_units.unique()
            for unit in unique_units:
                unit_mask = age_units == unit
                for route in route_names[unit_mask].unique():
                    route_mask = route_names == route
                    combined_mask = unit_mask & route_mask
                    ax.scatter(
                        x[combined_mask],
                        y[combined_mask],
                        label=f"{unit} - {route}",
                        marker=marker_styles.get(route, 'o')  # Default to 'o' if route not in dictionary
                    )
            
            # Add labels and legend
            ax.set_xlabel('Check In Age')
            ax.set_ylabel('Haze (%)')
            ax.legend(title='Age Unit and Route Name')
            ax.set_title('Haze Check-In')

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
            route_names = df['route_name']  # Assuming this is the column with route names

            # Create the plot
            fig, ax = plt.subplots()

            unique_units = age_units.unique()
            for unit in unique_units:
                unit_mask = age_units == unit
                for route in route_names[unit_mask].unique():
                    route_mask = route_names == route
                    combined_mask = unit_mask & route_mask
                    ax.scatter(
                        x[combined_mask],
                        y[combined_mask],
                        label=f"{unit} - {route}",
                        marker=marker_styles.get(route, 'o')  # Default to 'o' if route not in dictionary
                    )

            # Add labels and legend
            ax.set_xlabel('Check In Age')
            ax.set_ylabel('Weight (g)')
            ax.legend(title='Age Unit and Route Name')
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
            route_names = df['route_name']  # Assuming this is the column with route names

            # Create the plot
            fig, ax = plt.subplots()

            unique_units = age_units.unique()
            for unit in unique_units:
                unit_mask = age_units == unit
                for route in route_names[unit_mask].unique():
                    route_mask = route_names == route
                    combined_mask = unit_mask & route_mask
                    ax.scatter(
                        x[combined_mask],
                        y[combined_mask],
                        label=f"{unit} - {route}",
                        marker=marker_styles.get(route, 'o')  # Default to 'o' if route not in dictionary
                    )

            # Add labels and legend
            ax.set_xlabel('Check In Age')
            ax.set_ylabel('Mesh Width (um)')
            ax.legend(title='Age Unit and Route Name')
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
            route_names = df['route_name']  # Assuming this is the column with route names

            # Create the plot
            fig, ax = plt.subplots()

            unique_units = age_units.unique()
            for unit in unique_units:
                unit_mask = age_units == unit
                for route in route_names[unit_mask].unique():
                    route_mask = route_names == route
                    combined_mask = unit_mask & route_mask
                    ax.scatter(
                        x[combined_mask],
                        y[combined_mask],
                        label=f"{unit} - {route}",
                        marker=marker_styles.get(route, 'o')  # Default to 'o' if route not in dictionary
                    )

            # Add labels and legend
            ax.set_xlabel('Check In Age')
            ax.set_ylabel('Internal Resistance (Ohm)')
            ax.legend(title='Age Unit and Route Name')
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

# OLD STATIC FUNCTION 9/25/24
def create_static_jmp_panel(ec_optics_df):
    y_variables = ['coulombic_efficiency', 'tint_max_current', 'tint_charge_a', 'tint_charge_b',
                'tint_charge_c', 'tint_max_current_time', 'delta_initial_final_percentage',
                'delta_max_min_percentage', 'final_percentage', 'initial_percentage',
                'max_percentage', 'min_percentage', 'tint_ten_time', 'tint_five_time',
                'a_initial', 'b_initial', 'deltaE_initial', 'deltaE_final',
                'mesh_width_checkin', 'tint_time_eighty_vlt',
                'tint_ten_b', 'tint_ten_a']
    # Remove y variables without data in the dataframe
    y_variables = [y for y in y_variables if y in ec_optics_df.columns and ec_optics_df[y].notna().sum() > 0]
    cycle_plots = {}

    # Define marker shapes for each route_name
    marker_shapes = {
        'Ambient': 'o',  # Circle
        'Demo': 's',  # Square
        'Oven': '^',  # Triangle
        'Samples': 'D',  # Diamond
        'Weatherometer': 'x',} # X

    for y_variable in y_variables:
        x = ec_optics_df['cycle_number'].values
        x = x.astype(np.int64)
        y = ec_optics_df[y_variable].values
        y = y.astype(np.int64)
        fig, ax = plt.subplots(figsize=(10, 6))
    
        # Plot data points with different shapes based on route_name
        for route_name, marker_shape in marker_shapes.items():
            mask = ec_optics_df['route_name'] == route_name
            ax.scatter(x[mask], y[mask], color='black', marker=marker_shape, label=route_name, s=300) 
        
        ax.set_xticks(np.arange(0, 200, 100))
        ax.set_xlim(-50, 200)
        
        # Add labels
        ax.set_xlabel('Cycle #')
        ax.set_ylabel(y_variable)
        ax.set_title(f'{y_variable} vs Cycle #')
        ax.legend(title='Route Name')

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
    
        # Boxplot with light grey fill
        boxprops = dict(facecolor='lightgrey', color='black')  # Light grey fill
        whiskerprops = dict(color='black')
        capprops = dict(color='black')

        ax.boxplot(box_data, positions=range(1, len(unique_x) + 1), widths=0.5, patch_artist=True, 
                labels=[str(val) for val in unique_x], boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops)

        if not ec_optics_df.empty:
            avg_y = [np.mean([y[i] for i in range(len(x)) if x[i] == val]) for val in unique_x]
            table_data = list(zip(unique_x, avg_y))
            table = plt.table(cellText=table_data, colLabels=['Cycle #', 'Average'], cellLoc='center', loc='bottom', bbox=[0.1, -0.3, 0.8, 0.2])

        plot_filename = f"{y_variable}.png"
        fig.savefig(plot_filename, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free memory

        cycle_plots[y_variable] = plot_filename

    image_panes = [pn.pane.PNG(filename, width=600, height=600) for filename in cycle_plots.values()]
    cycle_jmp_layout = pn.GridBox(*image_panes, ncols=2, sizing_mode='stretch_width')

    return cycle_jmp_layout



def create_interactive_jmp_panel_old(ec_optics_df):
    # Define route selector
    route_values = ['Plot All'] + ec_optics_df['route_name'].unique().tolist()
    route_select = pn.widgets.Select(name='Select Route', options=route_values, value=route_values[0])

    # Function to filter DataFrame based on the selected route
    def filter_dataframe(route_name):
        if route_name == 'Plot All':
            return ec_optics_df  # Use the whole DataFrame
        else:
            return ec_optics_df[ec_optics_df['route_name'] == route_name]  # Filter by selected route_name

    # Function to create a row for each y-value
    def create_row(y_value):
        # Get the initial filtered DataFrame
        filtered_df = filter_dataframe(route_select.value)

        # Get the slider range from the y_value_ranges dictionary
        slider_range = y_value_ranges.get(y_value, (-100, 100))

        # Create a RangeSlider for the y-value
        range_slider = pn.widgets.RangeSlider(name=f'Select Range for {y_value}', start=slider_range[0], end=slider_range[1], value=slider_range)

        # Function to calculate the percentage of values within the slider range
        def calculate_percentage(slider_value):
            within_range = filtered_df[(filtered_df[y_value] >= slider_value[0]) & (filtered_df[y_value] <= slider_value[1])]
            total_values = len(filtered_df)
            
            if total_values == 0:
                return "No values in the selected range."
            
            percentage = (len(within_range) / total_values) * 100
            return f"{percentage:.2f}% of values are within the selected range."

        # Dynamic Markdown pane for percentage display
        percentage_display = pn.pane.Markdown(calculate_percentage(range_slider.value))

        # Update the percentage display when the slider changes
        @pn.depends(range_slider.param.value, watch=True)
        def update_percentage(slider_value):
            filtered_df = filter_dataframe(route_select.value)
            percentage_display.object = calculate_percentage(slider_value)

        # Create the plot
        plot = make_jmp_box(filtered_df, 'cycle_number', y_value, 'device_id', y_value)

        # Update the plot when the route selection changes
        @pn.depends(route_select.param.value, watch=True)
        def update_plot(route):
            filtered_df = filter_dataframe(route)
            plot.object = make_jmp_box(filtered_df, 'cycle_number', y_value, 'device_id', y_value)

        # Combine everything into a row
        row = pn.Column(range_slider, percentage_display, plot)
        return row

    # Create a layout with the route selector at the top
    layout = pn.Column(route_select)

    # Create a row for each y-value
    for y_value in y_value_ranges.keys():
        layout.append(create_row(y_value))

    return layout

# Example dictionary of y-values with corresponding slider ranges
y_value_ranges = {
    'coulombic_efficiency': (0, 200), 
    'bleach_final_current': (-10, 50),
    'tint_time_eighty_vlt': (0, 200),
    'deltaE_initial': (0,20),
    'final_percentage': (0, 100), 
    'initial_percentage': (0, 100),
    'min_percentage': (0, 100),
    'a_initial': (-30,30), 
    'tint_ten_a': (-30,30), 
    'b_initial': (-30,30), 
    'tint_ten_b': (-30,30), 
    # Add more y-values as needed...
}

def create_interactive_jmp_panel(ec_optics_df):
    # Define route selector
    route_values = ['Plot All'] + ec_optics_df['route_name'].unique().tolist()
    route_select = pn.widgets.Select(name='Select Route', options=route_values, value=route_values[0])

    # Function to filter DataFrame based on the selected route
    def filter_dataframe(route_name):
        if route_name == 'Plot All':
            return ec_optics_df  # Use the whole DataFrame
        else:
            return ec_optics_df[ec_optics_df['route_name'] == route_name]  # Filter by selected route_name

    # Function to create a row for each y-value
    def create_row(y_value):
        # Get the initial filtered DataFrame
        filtered_df = filter_dataframe(route_select.value)

        # Get the slider range from the y_value_ranges dictionary
        slider_range = y_value_ranges.get(y_value, (-100, 100))

        # Create a RangeSlider for the y-value
        range_slider = pn.widgets.RangeSlider(name=f'Select Range for {y_value}', start=slider_range[0], end=slider_range[1], value=slider_range)

        # Function to calculate the percentage of values within the slider range
        def calculate_percentage(slider_value):
            within_range = filtered_df[(filtered_df[y_value] >= slider_value[0]) & (filtered_df[y_value] <= slider_value[1])]
            total_values = len(filtered_df)
            
            if total_values == 0:
                return "No values in the selected range.", None, None
            
            percentage = (len(within_range) / total_values) * 100

            # Determine icon color
            if percentage == 100:
                icon = "✅"  # Green checkmark
            elif percentage > 0:
                icon = "⚠️"  # Yellow warning sign
            else:
                icon = "❌"  # Red cross

            # Find highest cycle number and corresponding device name
            if not within_range.empty:
                max_cycle_row = within_range.loc[within_range['cycle_number'].idxmax()]
                highest_cycle = max_cycle_row['cycle_number']
                device_name = max_cycle_row['device_id']
            else:
                highest_cycle, device_name = None, None

            output_text = (
                f"Percentage within range: {percentage:.2f}% {icon}\n"
                f"Highest cycle number: {highest_cycle} (Device: {device_name})\n"
                "Average cycle number at failure: "
            )

            return output_text, percentage, icon

        # Dynamic Markdown pane for percentage display
        percentage_display = pn.pane.Markdown(calculate_percentage(range_slider.value)[0])

        # Update the percentage display when the slider changes
        @pn.depends(range_slider.param.value, watch=True)
        def update_percentage(slider_value):
            filtered_df = filter_dataframe(route_select.value)
            percentage_display.object, _, _ = calculate_percentage(slider_value)

        # Create the plot
        plot = make_jmp_box(filtered_df, 'cycle_number', y_value, 'device_id', y_value)

        # Update the plot when the route selection changes
        @pn.depends(route_select.param.value, watch=True)
        def update_plot(route):
            filtered_df = filter_dataframe(route)
            plot.object = make_jmp_box(filtered_df, 'cycle_number', y_value, 'device_id', y_value)

        # Combine everything into a row
        row = pn.Column(range_slider, percentage_display, plot)
        return row

    # Create a layout with the route selector at the top
    layout = pn.Column(route_select)

    # Create a row for each y-value
    for y_value in y_value_ranges.keys():
        layout.append(create_row(y_value))

    return layout



# TESTING AREA
def testing_area():
    ec_optics_df = pd.read_csv('ec_optics_df.csv')
    ec_optics_df['coulombic_efficiency'] *= 100
    print(ec_optics_df)
    print(ec_optics_df.columns)
    #panel_layout = create_interactive_jmp_panel(ec_optics_df)
    panel_layout = create_interactive_jmp_panel(ec_optics_df)
    #panel_layout.servable()
    template = pn.template.FastListTemplate(
            title='Baseline Reporting Dashboard',
            main=panel_layout,
            accent_base_color="#00564a",
            header_background="#00564a",
        )
    template.show()





' ########### END JMP LAYOUT ####################'



def make_shaded_line_plot(x, y, title):
    # Create a HoloViews Curve (line plot)
    line_plot = hv.Curve((x, y), kdims='x', vdims='y')

    # Create a HoloViews Area (shaded region under the curve)
    # Fill from zero to the curve
    shaded_area = hv.Area((np.concatenate([x, x[::-1]]), np.concatenate([np.zeros_like(y), y[::-1]])),
                        kdims='x', vdims='y').opts(
        fill_color='lightgreen',  # Light green color
        fill_alpha=0.3,  # Mostly transparent
        line_color=None  # No line border
    )

    # Overlay the shaded area and the line plot
    plot = shaded_area * line_plot

    # Apply options for styling
    plot.opts(
        opts.Curve(title=title, color='green', line_width=2),  # Line style
        opts.Overlay(legend_position='top_left')  # Position the legend
    )

    return plot







def make_shaded_scatter_plot(x, y, title, y_string):
    if len(x) == 0 or len(y) == 0:
        return "Data is empty. No plot to display."

    # Create a HoloViews Scatter (plot with no connecting lines)
    scatter_plot = hv.Scatter((x, y), kdims='x', vdims='y')
    
    # Create a HoloViews Area (shaded region under the scatter points)
    shaded_area = hv.Area((x, np.zeros_like(y)), kdims='x', vdims='y').opts(
        fill_color='lightgreen',  # Light green color
        fill_alpha=0.3,  # Mostly transparent
        line_color=None  # No line border
    )
    
    # Calculate y-axis bounds
    y_min, y_max = min(y), max(y)
    y_range = (y_min - 5, y_max + 5)  # Extend the y-axis range by 5 units

    # Overlay the shaded area and the scatter plot
    plot = shaded_area * scatter_plot
    
    # Apply options for styling including the title and y-axis bounds
    plot.opts(
        opts.Scatter(
            color='green', 
            size=5, 
            title=title, 
            xlabel='Cycle', 
            ylabel=y_string, 
            ylim=y_range,  # Set the y-axis bounds directly
            width=800,
            height=400  # Ensure height is also set
        ),
        opts.Overlay(
            legend_position='top_left'  # Position the legend
        )
    )
    
    return plot

def update_plot(device_id, y_string, df):
    # Filter DataFrame based on the selected device ID
    filtered_df = df[df['device_id'] == device_id]
    
    # Check if filtered DataFrame is empty
    if filtered_df.empty:
        return f"No data available for device ID: {device_id}"
    
    # Extract data from filtered DataFrame
    x = filtered_df['cycle'].values
    y = filtered_df[y_string].values
    title = f'{y_string.replace("_", " ").title()} Over Cycle for Device ID: {device_id}'
    
    # Create and return the plot
    return make_shaded_scatter_plot(x, y, title, y_string)

def create_panel_plot(df, y_string):
    # Create a select widget for device IDs
    device_ids = df['device_id'].unique()
    select = pn.widgets.Select(name='Device ID', options=list(device_ids))
    
    # Define a callback function to update the plot based on the selected device ID
    @pn.depends(device_id=select)
    def plot_callback(device_id):
        if device_id is None:
            return "Please select a device ID."
        return update_plot(device_id, y_string, df)

    # Create and return the Panel layout with the select widget and the plot
    layout = pn.Column(select, plot_callback)
    return layout

' ######### ALTERNATE ARBIN PLOTS """"""""""""'
import panel as pn
import holoviews as hv
import numpy as np
import pandas as pd


def make_shaded_scatter_plot(x, y, title, y_string):
    if len(x) == 0 or len(y) == 0:
        return "Data is empty. No plot to display."

    scatter_plot = hv.Scatter((x, y), kdims='x', vdims='y')
    shaded_area = hv.Area((x, np.zeros_like(y)), kdims='x', vdims='y').opts(
        fill_color='lightgreen',
        fill_alpha=0.3,
        line_color=None
    )
    y_min, y_max = min(y), max(y)
    y_range = (y_min - 5, y_max + 5)

    plot = shaded_area * scatter_plot
    plot.opts(
        opts.Scatter(
            color='green',
            size=5,
            title=title,
            xlabel='Cycle',
            ylabel=y_string,
            ylim=y_range,
            width=800
        ),
        opts.Overlay(
            legend_position='top_left'
        )
    )
    
    return plot

def update_plot(df, device_id, y_string):
    filtered_df = df[df['device_id'] == device_id]
    
    if filtered_df.empty:
        return f"No data available for device ID: {device_id}"
    
    x = filtered_df['cycle'].values
    y = filtered_df[y_string].values
    title = f'{y_string.replace("_", " ").title()} Over Cycle for Device ID: {device_id}'
    
    return make_shaded_scatter_plot(x, y, title, y_string)

def create_single_panel_plot(df):
    if not df.empty: # if not empty
        device_ids = df['device_id'].unique()
        y_strings = ['coulombic_efficiency', 'bleach_internal_resistance', 'tint_internal_resistance', 'delta_tint_bleach_internal_resistance',
                    'bleach_charge', 'max_tint_current', 'tint_charge', 'total_bleach_time', 'total_tint_time']
        
        device_select = pn.widgets.Select(name='Device ID', options=list(device_ids))
        y_select = pn.widgets.Select(name='Y-String', options=y_strings)
        
        @pn.depends(device_id=device_select, y_string=y_select)
        def plot_callback(device_id, y_string):
            if device_id is None or y_string is None:
                return "Please select both device ID and y-string."
            plot = update_plot(df, device_id, y_string)
            if isinstance(plot, str):
                return plot
            return plot

        layout = pn.Column(device_select, y_select, plot_callback)
    else: layout = pn.Column('No Arbin cycling data found in database')
    return layout

'#############################################'


def extract_unique_parent_dirs(path_list):
    unique_dirs = set()
    
    for path in path_list:
        # Normalize the path for the OS
        normalized_path = os.path.normpath(path)
        
        # Split the path into components
        path_parts = normalized_path.split(os.sep)
        
        # Find the index of 'cycle' or 'precycle'
        cycle_index = next((i for i, part in enumerate(path_parts) if 'cycle' in part or 'precycle' in part), None)
        
        if cycle_index is not None:
            # Rebuild the path up to the index before 'cycle' or 'precycle'
            parent_dir = os.path.join(*path_parts[:cycle_index])
            
            # Ensure the path is properly formatted with leading '/' if the original path was absolute
            if os.path.isabs(path):
                parent_dir = os.path.join(os.sep, parent_dir)
            
            # Add trailing '/' if not already present
            if not parent_dir.endswith(os.sep):
                parent_dir += os.sep
            
            parent_dir = clean_path(parent_dir)
            unique_dirs.add(parent_dir)
    
    return sorted(unique_dirs)


def find_folders_recursively(path):
    folder_list = []
    local_list = os.listdir(path)
    
    for item in local_list:
        item_path = os.path.join(path, item)
        
        if os.path.isdir(item_path):
            # Add directory to the list
            folder_list.append(item_path)
            # Recursively find subdirectories
            item_path = clean_path(item_path)
            folder_list += find_folders_recursively(item_path)
    
    return folder_list

def find_photo_folder_paths(path_list):
    photo_folder_paths = []
    for path in path_list:
        if 'pictures' in path:
            path = clean_path(path)
            photo_folder_paths.append(path)
    return photo_folder_paths

def find_pictures_subdirectories(directory_path):
    pictures_dirs = []
    for root, dirs, files in os.walk(directory_path):
        for dir_name in dirs:
            if 'pictures' in dir_name.lower():
                pictures_dirs.append(clean_path(os.path.join(root, dir_name)))
    return pictures_dirs

def find_warmup_paths(path_list):
    photo_folder_paths = []
    for path in path_list:
        path = str(path) # convert from PosizPath object to string
        if '/precycle' in path or '\\precycle' in path:
            photo_folder_paths.append(clean_path(path))
    return photo_folder_paths

def find_cycle_paths(path_list):
    photo_folder_paths = []
    for path in path_list:
        path_str = str(path)
        if '/cycle' in path_str or '\\cycle' in path_str:
            photo_folder_paths.append(clean_path(path))
    return photo_folder_paths

def get_corresp_ec_filepaths(path_list):
    ec_paths = []
    for path in path_list:
        directory = os.path.dirname(path)
        items = os.listdir(directory)
        # should only be one .tyntEC file
        for item in items:
            if 'tyntEC' in item:
                ec_file_path = os.path.join(directory, item)
                ec_paths.append(clean_path(ec_file_path))
    return ec_paths


def get_all_raw_data(local_paths):
    # Define the column names
    columns = ['Time', 'Voltage (V)', 'Current (mA)', 'Charge (C)', 'Step Number',
        'Programmed Voltage (V)', 'Programmed Current (A)', 'Control Mode',
        'id', 'cycle']
    # Create an empty DataFrame with the specified columns
    final_df = pd.DataFrame(columns=columns)
    for path in local_paths:
        ### NEED TO ADD CATCH IF GOOGLE DRIVE FILES (not just folders) ARE DELETED OR NOT CONNECTED PROPERLY!
        try: 
            print(path.split(os.sep))
            file = path.split(os.sep)[-1]
            file = file.split('.')[0]
            id = int(file.split('_')[2])
            pattern = r'\d+'
            cycle_string = file.split('_')[3]
            print(cycle_string)
            cycle = re.findall(pattern, cycle_string)
            cycle = int(cycle[0])
            raw_ec_cycle_data = pd.read_csv(str(path), sep='\t', comment='#', index_col=0)
            time = pd.to_datetime(raw_ec_cycle_data['Time'], format = '%Y-%m-%d %H:%M:%S.%f')
            startTime = time[0]
            time -= startTime
            time = time.dt.total_seconds()
            voltage = raw_ec_cycle_data['Voltage (V)']
            current = raw_ec_cycle_data['Current (A)']
            current = current*1000 # convert from A to mA
            raw_ec_cycle_data['Current (mA)'] = current
            del raw_ec_cycle_data['Current (A)']
            charge = raw_ec_cycle_data['Charge (C)']
            raw_ec_cycle_data['Time'] = time
            raw_ec_cycle_data['id'] = [id]*len(time)
            raw_ec_cycle_data['cycle'] = [cycle]*len(time)

            print(raw_ec_cycle_data.columns)
            final_df = pd.concat([final_df, raw_ec_cycle_data], ignore_index=True)

        except Exception:
            print('\n Error: Unable to locate EC file on drive')
    return final_df


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import panel as pn

def correlation_heatmap_old(df):
    data = {
        'A': [1, 2, 3, 4, 5],
        'B': [5, 6, 7, 8, 9],
        'C': [9, 10, 11, 12, 13],
        'D': [1, 3, 5, 7, 9]
    }
    df = pd.DataFrame(data)
    correlation_matrix = df.corr()

    # Create the heatmap using seaborn
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)

    # Add a title
    ax.set_title('Correlation Heatmap')

    # Return the Matplotlib figure embedded in Panel
    return pn.pane.Matplotlib(fig, tight=True)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import panel as pn

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import panel as pn

# Ensure Panel extension for Matplotlib integration
pn.extension('matplotlib')

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import panel as pn

# Ensure Panel extension for Matplotlib integration
pn.extension('matplotlib')



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import panel as pn
import numpy as np

pn.extension('matplotlib')

def correlation_heatmap(df):
    # Define the variables for correlation and slope calculation
    variables = [
        'bleach_final_current', 'bleach_max_current', 'tint_final_current', 'tint_max_current',
        'tint_charge_a', 'tint_charge_b', 'tint_charge_c', 'tint_charge_d', 'charge_in', 
        'charge_out', 'tint_max_current_time', 'bleach_ten_time', 'bleach_five_time', 
        'bleach_one_time', 'bleach_point_one_time', 'delta_initial_final_percentage', 
        'delta_max_min_percentage', 'final_percentage', 'initial_percentage', 
        'max_percentage', 'min_percentage', 'tint_ten_time', 'tint_five_time', 
        'tint_one_time', 'tint_point_one_time', 'tint_ten_a', 'tint_five_a', 
        'tint_one_a', 'tint_point_one_a', 'a_final', 'a_initial', 'a_max', 'a_min', 
        'tint_ten_b', 'tint_five_b', 'tint_one_b', 'tint_point_one_b', 'b_final', 
        'b_initial', 'b_max', 'b_min', 'deltaE_initial', 'deltaE_final', 
        'tint_five_deltaE', 'deltaE_min', 'tint_one_deltaE', 'tint_point_one_deltaE', 
        'tint_ten_deltaE', 'tint_time_eighty_vlt'
    ]

    # Prepare lists to hold slopes and last cycle absolute values
    slopes = {var: [] for var in variables}
    last_cycle_abs = {var: [] for var in variables}

    # Iterate over each device_id
    for device_id in df['device_id'].unique():
        print(f"\nProcessing device: {device_id}")

        # Filter the DataFrame for this device_id
        device_df = df[df['device_id'] == device_id].sort_values('cycle_number')

        # Get the unique cycle numbers and check if there are at least 3
        unique_cycles = device_df['cycle_number'].unique()
        if len(unique_cycles) < 3:
            print(f"Not enough data for device {device_id}. Skipping...")
            continue

        # Select the first two and the last cycle
        first_cycle = unique_cycles[0]
        second_cycle = unique_cycles[1]
        last_cycle = unique_cycles[-1]

        # Get the data for the selected cycles
        first_cycle_data = device_df[device_df['cycle_number'] == first_cycle]
        second_cycle_data = device_df[device_df['cycle_number'] == second_cycle]
        last_cycle_data = device_df[device_df['cycle_number'] == last_cycle]

        # Calculate the slope between the first and second cycle for each variable
        for var in variables:
            if var in device_df.columns:
                # Calculate the slope between the first and second cycle
                if not first_cycle_data[var].empty and not second_cycle_data[var].empty:
                    slope = (second_cycle_data[var].values[0] - first_cycle_data[var].values[0]) / (second_cycle - first_cycle)
                    slopes[var].append(slope)
                
                # Get the absolute value for the last cycle
                if not last_cycle_data[var].empty:
                    last_cycle_abs_value = abs(last_cycle_data[var].values[0])
                    last_cycle_abs[var].append(last_cycle_abs_value)

    # Convert the lists of slopes and last cycle absolute values to DataFrames
    slope_df = pd.DataFrame(slopes)
    last_cycle_abs_df = pd.DataFrame(last_cycle_abs)

    # Print the slope values for all devices
    print("\nSlope values across all devices:")
    print(slope_df)

    # Print the absolute values for the last cycle across all devices
    print("\nLast cycle absolute values across all devices:")
    print(last_cycle_abs_df)

    # Create a correlation DataFrame
    correlation_results = {}
    
    for var in variables:
        if var in slope_df.columns and var in last_cycle_abs_df.columns:
            # Calculate correlation for the current variable
            correlation = slope_df[var].corr(last_cycle_abs_df[var])
            correlation_results[var] = correlation

    # Create a DataFrame for correlations
    correlation_df = pd.DataFrame(correlation_results, index=['Correlation']).T

    # Print the correlation matrix
    print("\nCorrelation matrix:")
    print(correlation_df)

    # Create the heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(correlation_df, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)

    # Add a title to the heatmap
    ax.set_title(f'Correlation Heatmap: Slope vs Last Cycle Absolute Values')

    # Return the Matplotlib figure embedded in Panel
    return pn.pane.Matplotlib(fig, tight=True)

# Example usage:
# correlation_heatmap(df)


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import panel as pn

pn.extension()




def correlation_heatmap_with_slope_old(df):
    # Define the variables for correlation and slope calculation
    variables = [
        #'bleach_max_current', 'tint_final_current', 'tint_max_current',
        'tint_charge_a', 'tint_charge_b', 'tint_charge_c', 'tint_charge_d', 'charge_in', 
        'charge_out', 'tint_max_current_time', 'bleach_ten_time', 'bleach_five_time', 
        'delta_max_min_percentage', 'final_percentage', 'initial_percentage', 
        'max_percentage', 'min_percentage', 'tint_ten_time', 'tint_five_time', 
        'tint_one_time', 'tint_point_one_time', 'tint_ten_a', 
        'a_initial', 'tint_ten_b', 'b_initial', 'tint_time_eighty_vlt'
    ]

    # Collecting device, route, and cycle information
    devices_used = []
    routes_used = []  # If there is a column for route in the DataFrame, add to this
    cycles_used = []

    # Make a copy of the dataframe to avoid modifying the original
    df_copy = df.copy()

    # Iterate over each device_id
    for device_id in df['device_id'].unique():
        print(f"\nProcessing device: {device_id}")
        devices_used.append(device_id)

        # Filter the DataFrame for this device_id
        device_df = df[df['device_id'] == device_id].sort_values('cycle_number')

        # Get the unique cycle numbers and check if there are at least 3
        unique_cycles = device_df['cycle_number'].unique()
        print('Unique cycles', unique_cycles)
        if len(unique_cycles) < 3:
            print(f"Not enough data for device {device_id}. Skipping...")
            continue

        # Select the first two and the last cycle
        first_cycle = unique_cycles[0]
        second_cycle = unique_cycles[1]
        last_cycle = unique_cycles[-1]

        cycles_used.append((first_cycle, second_cycle, last_cycle))

        # Get the data for the selected cycles
        first_cycle_data = device_df[device_df['cycle_number'] == first_cycle]
        second_cycle_data = device_df[device_df['cycle_number'] == second_cycle]
        last_cycle_data = device_df[device_df['cycle_number'] == last_cycle]

        # Calculate the slope between the first and second cycle for each variable
        for var in variables:
            if var in device_df.columns:
                # Calculate the slope between the first and second cycle
                if not first_cycle_data[var].empty and not second_cycle_data[var].empty:
                    slope = (second_cycle_data[var].values[0] - first_cycle_data[var].values[0]) / (second_cycle - first_cycle)

                    # Add the slope value to the dataframe for the last cycle
                    df_copy.loc[(df_copy['device_id'] == device_id) & (df_copy['cycle_number'] == last_cycle), f'{var}_slope'] = slope

    # Filter the DataFrame to keep only the last cycle data
    last_cycle_df = df_copy.groupby('device_id').apply(lambda group: group[group['cycle_number'] == group['cycle_number'].max()])

    # Prepare data for correlation matrix
    columns_to_correlate = [f'{var}_slope' for var in variables if f'{var}_slope' in last_cycle_df.columns]
    columns_to_correlate += [var for var in variables if var in last_cycle_df.columns]

    # Calculate the correlation matrix
    correlation_matrix = last_cycle_df[columns_to_correlate].corr()

    # Filter the correlation matrix based on selected variables
    x_variables = [f'{var}_slope' for var in variables]
    y_variables = [
        'delta_max_min_percentage', 'final_percentage', 'initial_percentage', 
        'max_percentage', 'min_percentage', 'tint_ten_time', 'tint_five_time', 
        'tint_one_time',  'tint_ten_a', 
        'a_initial', 'tint_ten_b', 'b_initial', 'tint_time_eighty_vlt']
    correlation_matrix = correlation_matrix.loc[y_variables, x_variables]
    
    # Find the top 5 strongest correlations
    strongest_correlations = correlation_matrix.unstack().abs().sort_values(ascending=False)
    strongest_correlations = strongest_correlations[~strongest_correlations.index.duplicated(keep='first')].head(5)

    print("\nTop 5 Strongest Correlations:")
    for (row_var, col_var), corr_value in strongest_correlations.items():
        print(f"Correlation between {row_var} and {col_var}: {corr_value:.4f}")
    
    # Collecting device, route, and cycle information for the summary
    summary_string = f"Devices used: {', '.join(map(str, devices_used))}\n"
    summary_string += f"Cycles used (First, Second, Last): {cycles_used}\n"
    
    # Print the correlation matrix
    print("\nCorrelation Matrix:")
    print(correlation_matrix)

    # Extract raw data for top 5 strongest correlations
    raw_data_tables = []
    for (row_var, col_var), corr_value in strongest_correlations.items():
        # Extract relevant data for row_var and col_var
        if row_var in last_cycle_df.columns and col_var in last_cycle_df.columns:
            raw_data = last_cycle_df[[row_var, col_var]]
            raw_data_table = pn.pane.DataFrame(raw_data, name=f"Raw Data for {row_var} and {col_var} (Correlation: {corr_value:.4f})")
            raw_data_tables.append(raw_data_table)
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    #mask |= np.eye(correlation_matrix.shape[0], dtype=bool)  # Mask diagonal values

    sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax, square=True)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0, ha='right', fontsize=5)
    # Add a title to the heatmap
    ax.set_title(f'Correlation Heatmap: Slope vs Last Cycle Absolute Values')

    # Return the Matplotlib figure, summary string, and the raw data tables
    return pn.Column(pn.pane.Matplotlib(fig, tight=True), summary_string, *raw_data_tables)

# Example usage:
# layout = correlation_heatmap_with_slope(df)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import panel as pn

def correlation_heatmap_with_slope(df):
    # Define the variables for correlation and slope calculation
    variables = [
        #'bleach_max_current', 'tint_final_current', 'tint_max_current',
        'tint_charge_a', 'tint_charge_b', 'tint_charge_c', 'tint_charge_d', 'charge_in', 
        'charge_out', 'tint_max_current_time', 'bleach_ten_time', 'bleach_five_time', 
        'delta_max_min_percentage', 'final_percentage', 'initial_percentage', 
        'max_percentage', 'min_percentage', 'tint_ten_time', 'tint_five_time', 
        'tint_one_time', 'tint_point_one_time', 'tint_ten_a', 
        'a_initial', 'tint_ten_b', 'b_initial', 'tint_time_eighty_vlt'
    ]

    devices_used = []
    cycles_used = []

    df_copy = df.copy()

    for device_id in df['device_id'].unique():
        

        # Filter the DataFrame for this device_id
        device_df = df[df['device_id'] == device_id].sort_values('cycle_number')

        unique_cycles = device_df['cycle_number'].unique()
        if len(unique_cycles) < 3:
            continue
        devices_used.append(device_id)

        # Select the first two and the last cycle
        first_cycle = unique_cycles[0]
        second_cycle = unique_cycles[1]
        last_cycle = unique_cycles[-1]
        cycles_used.append((first_cycle, second_cycle, last_cycle))

        first_cycle_data = device_df[device_df['cycle_number'] == first_cycle]
        second_cycle_data = device_df[device_df['cycle_number'] == second_cycle]
        last_cycle_data = device_df[device_df['cycle_number'] == last_cycle]

        # Calculate slope between the first and second cycle for each variable
        for var in variables:
            if var in device_df.columns and not first_cycle_data[var].empty and not second_cycle_data[var].empty:
                slope = (second_cycle_data[var].values[0] - first_cycle_data[var].values[0]) / (second_cycle - first_cycle)
                df_copy.loc[(df_copy['device_id'] == device_id) & (df_copy['cycle_number'] == last_cycle), f'{var}_slope'] = slope

    # Filter to keep only the last cycle data
    last_cycle_df = df_copy.groupby('device_id').apply(lambda group: group[group['cycle_number'] == group['cycle_number'].max()])

    # Prepare data for correlation matrix
    columns_to_correlate = [f'{var}_slope' for var in variables if f'{var}_slope' in last_cycle_df.columns]
    columns_to_correlate += [var for var in variables if var in last_cycle_df.columns]

    correlation_matrix = last_cycle_df[columns_to_correlate].corr()

    x_variables = [f'{var}_slope' for var in variables]
    y_variables = [
        'delta_max_min_percentage', 'final_percentage', 'initial_percentage', 
        'max_percentage', 'min_percentage', 'tint_ten_time', 'tint_five_time', 
        'tint_one_time',  'tint_ten_a', 
        'a_initial', 'tint_ten_b', 'b_initial', 'tint_time_eighty_vlt']
    
    # Filter out missing x_variables
    x_variables_filtered = [var for var in x_variables if var in correlation_matrix.columns]
    # Filter out missing y_variables
    y_variables_filtered = [var for var in y_variables if var in correlation_matrix.index]

    # Now use the filtered lists
    correlation_matrix = correlation_matrix.loc[y_variables_filtered, x_variables_filtered]
    #correlation_matrix = correlation_matrix.loc[y_variables, x_variables]

    strongest_correlations = correlation_matrix.unstack().abs().sort_values(ascending=False).head(5)

    print("\nTop 5 Strongest Correlations:")
    for (row_var, col_var), corr_value in strongest_correlations.items():
        print(f"Correlation between {row_var} and {col_var}: {corr_value:.4f}")

    summary_string = f"Devices used: {', '.join(map(str, devices_used))}\n"
    summary_string += f"Cycles used (First, Second, Last): {cycles_used}\n"

    # Print the correlation matrix
    print("\nCorrelation Matrix:")
    print(correlation_matrix)

    # Extract raw data for top 5 strongest correlations
    raw_data_tables = []
    for (row_var, col_var), corr_value in strongest_correlations.items():
        # Extract relevant data for row_var and col_var
        if row_var in last_cycle_df.columns and col_var in last_cycle_df.columns:
            for device_id in last_cycle_df['device_id'].unique():
                device_last_cycle_data = last_cycle_df[last_cycle_df['device_id'] == device_id]
                device_first_cycle_data = df[(df['device_id'] == device_id) & (df['cycle_number'] == first_cycle)]
                device_second_cycle_data = df[(df['device_id'] == device_id) & (df['cycle_number'] == second_cycle)]

                # Combine first, second, and last cycle data into one table
                raw_data = pd.DataFrame({
                    'Device ID': [device_id] * 3,
                    'Cycle': [first_cycle, second_cycle, last_cycle],
                    row_var: [
                        device_first_cycle_data[row_var].values[0] if not device_first_cycle_data.empty and row_var in device_first_cycle_data.columns else None,
                        device_second_cycle_data[row_var].values[0] if not device_second_cycle_data.empty and row_var in device_second_cycle_data.columns else None,
                        device_last_cycle_data[row_var].values[0] if not device_last_cycle_data.empty and row_var in device_last_cycle_data.columns else None
                    ],
                    col_var: [
                        device_first_cycle_data[col_var].values[0] if not device_first_cycle_data.empty and col_var in device_first_cycle_data.columns else None,
                        device_second_cycle_data[col_var].values[0] if not device_second_cycle_data.empty and col_var in device_second_cycle_data.columns else None,
                        device_last_cycle_data[col_var].values[0] if not device_last_cycle_data.empty and col_var in device_last_cycle_data.columns else None
                    ]
                })

                # Create description string with device ID and correlation value
                description = f"Raw Data for Device {device_id} - {row_var} and {col_var} (Correlation: {corr_value:.4f})"
                raw_data_table = pn.pane.DataFrame(raw_data, name=description)
                raw_data_tables.append(raw_data_table)


    def plot_heatmap(correlation_matrix):
        if correlation_matrix.empty:
            # Return an empty message instead of plotting
            print("Correlation matrix is empty. No heatmap to plot.")
            return pn.pane.Markdown("### No Data Available for Correlation Heatmap")

        # Otherwise, plot the heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Use try-except to handle any potential issues during plotting
        try:
            sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0, ha='right', fontsize=5)
            ax.set_title(f'Correlation Heatmap: Slope vs Last Cycle Absolute Values')

            # Save the figure
            fig_path = "correlation_heatmap.png"
            plt.savefig(fig_path)
            plt.close(fig)  # Close the figure after saving to avoid memory issues

            # Return the plot as a Panel PNG pane
            return pn.pane.PNG(fig_path, width=600, height=400)
        except ValueError as e:
            print(f"Error plotting heatmap: {e}")
            return pn.pane.Markdown("### Error plotting heatmap. Please check the data.")

    heatmap_pane = plot_heatmap(correlation_matrix)

    return pn.Column(heatmap_pane, summary_string, *raw_data_tables)

# Example usage:
# layout = correlation_heatmap_with_slope(df)





#df = pd.read_csv('/Users/sarahpearce/Documents/dashboards/ec_optics_df.csv')
#print(df)

#layout = correlation_heatmap_with_slope(df)
#layout = correlation_heatmap(df)
#main_content = layout
#template = pn.template.FastListTemplate(
#       title='Baseline Reporting Dashboard',
#        main=main_content,
#        accent_base_color="#00564a",
#        header_background="#00564a",
  #  )
#template.show()
def search_trd_name(df, search_string):
    # Ensure 'trd_name' is treated as string
    df['trd_name'] = df['trd_name'].astype(str)
    
    # Filter rows where 'trd_name' exactly matches the search string
    filtered_df = df[df['trd_name'] == search_string]
    
    # Extract corresponding 'id' values
    ids = filtered_df['id'].tolist()
    
    return ids

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

def create_dynamic_baseline_and_trd_dialog(df, trds_df, conn, cursor):
    # Create the application and the main window
    app = QApplication([])
    window = QWidget()
    window.setWindowTitle("Select Experiment and Devices to Plot")

    # Create a layout for the main window
    main_layout = QVBoxLayout(window)

    # Label and experiment type dropdown
    experiment_label = QLabel("Experiment Type:")
    main_layout.addWidget(experiment_label)

    experiment_dropdown = QComboBox()
    experiment_dropdown.addItems(['TRD', 'Baseline'])  # Two options: TRD and Baseline
    main_layout.addWidget(experiment_dropdown)

    # Label and second dropdown (either TRD or Baseline versions)
    label = QLabel("Select a Version:")
    main_layout.addWidget(label)

    second_dropdown = QComboBox()
    main_layout.addWidget(second_dropdown)
    experiment_type = experiment_dropdown.currentText()
    # Function to update the second dropdown based on experiment type
    def update_second_dropdown():
        experiment_type = experiment_dropdown.currentText()
        second_dropdown.clear()
        if experiment_type == 'TRD':
            second_dropdown.addItems(reversed(trds_df['trd_name'].values))
        elif experiment_type == 'Baseline':
            second_dropdown.addItems(reversed(df['baseline_version'].values))

    # Connect experiment dropdown's change event to update second dropdown
    experiment_dropdown.currentTextChanged.connect(update_second_dropdown)

    # Create a scroll area for checkboxes
    scroll_area = QScrollArea()
    checkbox_widget = QWidget()
    checkbox_layout = QVBoxLayout(checkbox_widget)
    checkbox_widget.setLayout(checkbox_layout)
    scroll_area.setWidget(checkbox_widget)
    scroll_area.setWidgetResizable(True)
    main_layout.addWidget(scroll_area)

    # Function to update checkboxes based on the second dropdown selection
    def update_checkboxes():
        # Clear previous checkboxes
        for i in reversed(range(checkbox_layout.count())):
            checkbox_layout.itemAt(i).widget().setParent(None)

        category = second_dropdown.currentText()
        if experiment_dropdown.currentText() == 'TRD':
            matching_ids = search_trd_name(trds_df, category)  # Assuming you have a search function for TRD
            trd_id = matching_ids[0]
            trd_devices = get_trd_devices(conn, cursor, trd_id)
            device_list = trd_devices['device_name'].values  # Assuming trd_devices has 'device_name' column
        else:
            matching_ids = search_baseline_name(df, category)
            baseline_id = matching_ids[0]
            baseline_devices = get_baseline_devices(conn, cursor, baseline_id)
            device_list = baseline_devices['device_name'].values  # Assuming baseline_devices has 'device_name' column

        for device in device_list:
            checkbox = QCheckBox(device)
            checkbox.setChecked(True)
            checkbox_layout.addWidget(checkbox)

    # Connect second dropdown's change event to update the checkboxes
    second_dropdown.currentTextChanged.connect(update_checkboxes)

    # Initialize the second dropdown and checkboxes for the first time
    update_second_dropdown()
    update_checkboxes()

    # Variables to hold the selection
    selected_version = None
    selected_devices = []

    # Function to capture the selections
    def capture_selections():
        nonlocal selected_version, selected_devices
        selected_version = second_dropdown.currentText()
        selected_devices = [checkbox.text() for checkbox in checkbox_widget.findChildren(QCheckBox) if checkbox.isChecked()]
        print(f"Selected Version: {selected_version}")
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
    return selected_version, selected_devices, experiment_type
