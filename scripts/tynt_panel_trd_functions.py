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
import io

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


