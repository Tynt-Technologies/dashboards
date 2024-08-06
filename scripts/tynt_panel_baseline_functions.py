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

def search_shared_drives_data():
    # Define potential base paths to search (customize based on your setup)
    base_paths = [
        os.path.expanduser('~'),  # Home directory
        '/Volumes',               # Common mount points
        '/media',                 # Alternative common mount points
        '/mnt'                    # Alternative common mount points
    ]

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
    # Fix google drive's FUCK UP
    # Regular expression pattern to match a timestamp (e.g., (7:24:24))
    base_path = '/Users/sarahpearce/Library/CloudStorage/GoogleDrive-sarah@tynt.io/Shared drives/Data'
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
    time = pd.to_datetime(df['Time'], format = '%Y-%m-%d %H:%M:%S.%f')
    startTime = time[0]
    time -= startTime
    time = time.dt.total_seconds()
    current = df['Current (A)']
    current = current*1000 # convert from A to mA
    df['Current (mA)'] = current
    df['Time'] = time
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


    return formatted_schedule, p

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

import os

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
            unique_dirs.add(parent_dir)
    
    return list(unique_dirs)


def create_dashboard(baseline_version, warmup_gif_path):
    pn.config.theme = 'dark' 
    title_text = f"# Dashboard for {baseline_version}"

    warmup_image_pane = pn.pane.Image(warmup_gif_path, width=400, height=300)

    dynamic_title = f"# Dashboard for {baseline_version}"
    table_html = "<h1>Example Table</h1><p>This is a table.</p>"
    dashboard = pn.Column(
        pn.pane.HTML(f"<style>{table_html}</style>"),
        pn.pane.Markdown(
            dynamic_title, 
            style={
                'color': '#00564a',
                'font-family': 'BlinkMacSystemFont',  # Set font family
                'font-size': '24px',                # Set font size
                'font-weight': 'bold'               # Set font weight
            }
        ),
        pn.Spacer(height=20),
        # Arrange images in a row
        pn.Row(
            pn.pane.Markdown("## Example Photos", style={'color': '#90EE90'}),
            pn.pane.Markdown("This section shows clear state photos before warmup cycling."),
        ),
        pn.Row(warmup_image_pane),
        pn.Spacer(height=20),
        pn.pane.Markdown("### Additional Information", style={'color': '#e6ffe60'}),
        pn.pane.Markdown("Haven't decided what additional information goes here. Likely formulation details.")
    )

    return dashboard

# Example usage
file_path = '/Users/sarahpearce/Library/CloudStorage/GoogleDrive-sarah@tynt.io/Shared drives/Data/Devices/2024/07/20240703/20240703_PB_4172/precycle1/20240703_PB_4172_precycle1.tyntEC'
steps_text = get_warmup_schedule(file_path)
print(steps_text)



