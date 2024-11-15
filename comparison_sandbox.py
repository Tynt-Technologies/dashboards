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

" ################### TRD SPECIFIC PANEL FUNCTIONS ########################## "


def create_dynamic_comparison_dialog(all_baselines_df, all_trds_df, conn, cursor):
    # Create the application and the main window
    app = QApplication([])
    window = QWidget()
    window.setWindowTitle("Select Two Experiments to Compare Device Performance")

    # Create a layout for the main window
    main_layout = QVBoxLayout(window)

    # Checkbox group to select the type of comparison
    comparison_label = QLabel("Select the type of comparison:")
    main_layout.addWidget(comparison_label)

    comparison_checkbox = QComboBox()
    comparison_checkbox.addItems(['Baseline vs TRD', 'TRD vs TRD', 'Baseline vs Baseline'])
    main_layout.addWidget(comparison_checkbox)

    # Label and first dropdown
    label1 = QLabel("Select the first option:")
    main_layout.addWidget(label1)

    first_dropdown = QComboBox()
    main_layout.addWidget(first_dropdown)

    # Create a scroll area for the first set of checkboxes
    scroll_area_1 = QScrollArea()
    checkbox_widget_1 = QWidget()
    checkbox_layout_1 = QVBoxLayout(checkbox_widget_1)
    checkbox_widget_1.setLayout(checkbox_layout_1)
    scroll_area_1.setWidget(checkbox_widget_1)
    scroll_area_1.setWidgetResizable(True)
    main_layout.addWidget(scroll_area_1)

    # Label and second dropdown
    label2 = QLabel("Select the second option:")
    main_layout.addWidget(label2)

    second_dropdown = QComboBox()
    main_layout.addWidget(second_dropdown)

    # Create a scroll area for the second set of checkboxes
    scroll_area_2 = QScrollArea()
    checkbox_widget_2 = QWidget()
    checkbox_layout_2 = QVBoxLayout(checkbox_widget_2)
    checkbox_widget_2.setLayout(checkbox_layout_2)
    scroll_area_2.setWidget(checkbox_widget_2)
    scroll_area_2.setWidgetResizable(True)
    main_layout.addWidget(scroll_area_2)

    # Function to update dropdowns based on the comparison type
    def update_dropdowns():
        first_dropdown.clear()
        second_dropdown.clear()

        comparison_type = comparison_checkbox.currentText()
        if comparison_type == 'Baseline vs TRD':
            first_dropdown.addItems(reversed(all_baselines_df['baseline_version'].values))
            second_dropdown.addItems(reversed(all_trds_df['trd_name'].values))
        elif comparison_type == 'TRD vs TRD':
            first_dropdown.addItems(reversed(all_trds_df['trd_name'].values))
            second_dropdown.addItems(reversed(all_trds_df['trd_name'].values))
        elif comparison_type == 'Baseline vs Baseline':
            first_dropdown.addItems(reversed(all_baselines_df['baseline_version'].values))
            second_dropdown.addItems(reversed(all_baselines_df['baseline_version'].values))

        # Update checkboxes for both dropdowns
        update_checkboxes()  # Call to update checkboxes after dropdowns are populated

    # Update checkboxes based on the first dropdown selection
    def update_checkboxes():
        # Clear previous checkboxes in both layouts
        for i in reversed(range(checkbox_layout_1.count())):
            checkbox_layout_1.itemAt(i).widget().setParent(None)
        for i in reversed(range(checkbox_layout_2.count())):
            checkbox_layout_2.itemAt(i).widget().setParent(None)

        # Get the selected options
        category1 = first_dropdown.currentText()
        category2 = second_dropdown.currentText()
        comparison_type = comparison_checkbox.currentText()

        # First dropdown logic
        if 'Baseline' in comparison_type:
            matching_ids = search_baseline_name(all_baselines_df, category1)
            if not matching_ids:
                print(f"No matching IDs found for category: {category1}")
                return  # Exit the function if no matching IDs are found

            baseline_id = matching_ids[0]
            baseline_devices = get_baseline_devices(conn, cursor, baseline_id)
            device_list_1 = baseline_devices['device_name'].values
        else:
            matching_ids = search_trd_name(all_trds_df, category1)
            if not matching_ids:
                print(f"No matching IDs found for category: {category1}")
                return

            trd_id = matching_ids[0]
            trd_devices = get_trd_devices(conn, cursor, trd_id)
            device_list_1 = trd_devices['device_name'].values

        # Populate checkboxes for the first dropdown
        for device in device_list_1:
            checkbox = QCheckBox(device)
            checkbox.setChecked(True)
            checkbox_layout_1.addWidget(checkbox)

        # Second dropdown logic
        if 'Baseline' in comparison_type:
            matching_ids = search_baseline_name(all_baselines_df, category2)
            if not matching_ids:
                print(f"No matching IDs found for second category: {category2}")
                return

            baseline_id = matching_ids[0]
            baseline_devices = get_baseline_devices(conn, cursor, baseline_id)
            device_list_2 = baseline_devices['device_name'].values
        else:
            matching_ids = search_trd_name(all_trds_df, category2)
            if not matching_ids:
                print(f"No matching IDs found for second category: {category2}")
                return

            trd_id = matching_ids[0]
            trd_devices = get_trd_devices(conn, cursor, trd_id)
            device_list_2 = trd_devices['device_name'].values

        # Populate checkboxes for the second dropdown
        for device in device_list_2:
            checkbox = QCheckBox(device)
            checkbox.setChecked(True)
            checkbox_layout_2.addWidget(checkbox)

    # Connect the comparison type change and dropdown changes to the update functions
    comparison_checkbox.currentTextChanged.connect(update_dropdowns)
    first_dropdown.currentTextChanged.connect(update_checkboxes)
    second_dropdown.currentTextChanged.connect(update_checkboxes)  # Also update when second dropdown changes

    # Initialize the dropdowns and checkboxes for the first time
    update_dropdowns()

    # Variables to hold the selection
    selected_comparison_type = None
    selected_trd_name1 = None
    selected_trd_name2 = None
    selected_devices_1 = []
    selected_devices_2 = []

    # Function to capture the selections
    def capture_selections():
        nonlocal selected_comparison_type, selected_trd_name1, selected_trd_name2, selected_devices_1, selected_devices_2
        selected_comparison_type = comparison_checkbox.currentText()
        selected_trd_name1 = first_dropdown.currentText()
        selected_trd_name2 = second_dropdown.currentText()

        # Capture selected devices for both dropdowns separately
        selected_devices_1 = [checkbox.text() for checkbox in checkbox_widget_1.findChildren(QCheckBox) if checkbox.isChecked()]
        selected_devices_2 = [checkbox.text() for checkbox in checkbox_widget_2.findChildren(QCheckBox) if checkbox.isChecked()]

        print(f"Selected Comparison Type: {selected_comparison_type}")
        print(f"Selected Option 1: {selected_trd_name1}")
        print(f"Selected Option 2: {selected_trd_name2}")
        print(f"Selected Devices from Option 1: {selected_devices_1}")
        print(f"Selected Devices from Option 2: {selected_devices_2}")

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
    return selected_comparison_type, selected_trd_name1, selected_trd_name2, selected_devices_1, selected_devices_2




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
    selected_comparison_type, selected_name_1, selected_name_2, selected_devices_1, selected_devices_2 = create_dynamic_comparison_dialog(all_baselines_df, all_trds_df, conn, cursor)
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


    ' ############################### MAKE NON-CYCLE CHECKIN PLOTS ################################# '
    import matplotlib.pyplot as plt

    checkin_df_dict_1 = get_haze_weight_meshwidth_devicewidth_bubbles_ir_joined(conn, cursor, device_id_list_1)
    print(device_id_list_2)
    checkin_df_dict_2 = get_haze_weight_meshwidth_devicewidth_bubbles_ir_joined(conn, cursor, device_id_list_2)


    

if __name__ == "__main__":
    main()

