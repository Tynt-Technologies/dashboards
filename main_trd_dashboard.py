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
from scripts.tynt_panel_trd_functions import *

hv.extension('bokeh')
pn.extension()

" ################### TRD SPECIFIC PANEL FUNCTIONS ########################## "

" ######################################## START GUIS ########################################################## "
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
        scatter_plot = group.hvplot.scatter(size=3, x=x, y=y, label=str(name))
        scatter_plots1.append(scatter_plot)
        
        if len(data[color_col1].unique()) > 1: 
            # ISSUES WITH FIRST AND LAST PTS IN LINES CONNECTING WHEN ONLY HAVE ONE ITEM TO GROUP BY
            line_plot = group.hvplot.line(x=x, y=y, label=str(name))
            line_plots1.append(line_plot)

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


def create_dashboard(row1_inputs, row2_inputs, trd_name, trd_devices_df, photo_paths):
    """
    Create an interactive line plot with color based on a column.

    Parameters:
    - data: pandas DataFrame with the data to plot
    - x: column name for x-axis
    - y: column name for y-axis
    - color_col: column name for coloring the lines
    - title: title of the plot
    """
    pn.config.theme = 'dark' 

    # '/Users/sarahpearce/Library/CloudStorage/GoogleDrive-sarah@tynt.io/Shared drives/Data/Devices/2024/06/20240626/20240626_DB_4126/cycle0/pictures/20240626_DB_4126_cycle0_PhotoNumber-1_StepNumber-1_StepTime-le-1s.jpg',



    r1p1, r1p1_bokeh, r1p2, r1p2_bokeh = make_row_traces(row1_inputs)
    r2p1,  r2p2 = make_row_box(row2_inputs)

    custom_css = """
    .bk-root .bk.pn-Column {
        background-color: rgba(0, 128, 0, 0.3) !important;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
    }
    .bk-root .bk.pn-Row {
        background-color: rgba(0, 128, 0, 0.3) !important;
    }
    .pn-Markdown {
        color: #333;
        font-family: 'Helvetica, Arial, sans-serif';
    }
    h1, h2, h3 {
        color: #555;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
    }
    th, td {
        padding: 10px;
        text-align: left;
        border: 1px solid #ddd;
    }
    th {
        background-color: #f2f2f2;
    }
    tr:hover {background-color: #f5f5f5;}
    """

    title_text = f"# Dashboard for {trd_name}"

    # HTML Table with Device ID, Notes, and Shorted Status
    device_list = trd_devices_df['device_name'].values
    devices_text = "\n".join(device_list)
    # HTML Table with inline styling
    table_html = """
    <table style="width: 100%; border-collapse: collapse; margin: 20px 0; font-family:BlinkMacSystemFont; font-size: 12px; color: #FFFFFF;">
        <thead>
            <tr style="background-color: #00564a;">
                <th style="padding: 10px; text-align: left; border: 1px solid #ddd; font-weight: bold;">Device ID</th>
                <th style="padding: 10px; text-align: left; border: 1px solid #ddd; font-weight: bold;">Device Shorted?</th>
                <th style="padding: 10px; text-align: left; border: 1px solid #ddd; font-weight: bold;">Device Notes</th>
            </tr>
        </thead>
        <tbody>
    """
    for row in range(len(trd_devices_df)):
        table_html += f"""
        <tr style="background-color: #00564a;">
            <td style="padding: 10px; text-align: left; border: 1px solid #ddd;">{trd_devices_df['device_name'].iloc[row]}</td>
            <td style="padding: 10px; text-align: left; border: 1px solid #ddd;">{trd_devices_df['shorted'].iloc[row]}</td>
            <td style="padding: 10px; text-align: left; border: 1px solid #ddd;">{trd_devices_df['notes'].iloc[row]}</td>
        </tr>
        """
    table_html += """
        </tbody>
    </table>
    """
    ######## END INTRO TABLE HTML ############
    print(photo_paths)
    photo_checkin_crop_box = (450, 1200, 2400, 3300)
    image_panes = [pn.pane.Image(crop_image(path, photo_checkin_crop_box), width=400, height=300) for path in photo_paths]

    dynamic_title = f"# Dashboard for {trd_name}"
    dashboard = pn.Column(
        pn.pane.HTML(f"<style>{custom_css}</style>"),
        pn.pane.Markdown(
            dynamic_title, 
            style={
                'color': '#00564a',
                'font-family': 'BlinkMacSystemFont',  # Set font family
                'font-size': '24px',                # Set font size
                'font-weight': 'bold'               # Set font weight
            }
        ),
        pn.pane.HTML(table_html),
        pn.Row(
            pn.Column(
                pn.pane.Markdown("## Check-In Current Traces colored by Cycle", style={'color': '#90EE90'}),
                pn.pane.Markdown("This section shows raw echem data over time colored by cycle."),
                r1p1, 
                #title
            ),
            pn.Spacer(width=20),
            pn.Column(
                pn.pane.Markdown("## Check-In Current Traces colored by Device ID", style={'color': '#90EE90'}),
                pn.pane.Markdown("This section shows raw echem data over time colored by device."),
                r1p2, 
                #title
            )
        ),
        pn.Spacer(height=20),
        pn.Row(  # Duplicated row
            pn.Column(
                pn.pane.Markdown("## Check-In Max Current by Cycle", style={'color': '#90EE90'}),
                pn.pane.Markdown("This section pulls summary echem data from the database."),
                r2p1, 
                #title
            ),
            pn.Spacer(width=20),
            pn.Column(
                pn.pane.Markdown("## Check-In Max Current by Cycle", style={'color': '#90EE90'}),
                pn.pane.Markdown("This section pulls summary echem data from the database."),
                r2p2, 
                #title
            )
        ),
        pn.Spacer(height=20),
        # Arrange images in a row
        pn.Row(
            pn.pane.Markdown("## Example Photos", style={'color': '#90EE90'}),
            pn.pane.Markdown("This section shows clear state photos before warmup cycling."),
        ),
        pn.Row(*image_panes),
        pn.Spacer(height=20),
        pn.pane.Markdown("### Additional Information", style={'color': '#e6ffe60'}),
        pn.pane.Markdown("Haven't decided what additional information goes here. Likely formulation details.")
    )

    return dashboard

# Example usage (assuming you have a pandas DataFrame `df` with appropriate data):
# dashboard = create_plot(df, x='date', y='value', color_col='category', title='My Dashboard')
# dashboard.show()



# Example usage (assuming you have a pandas DataFrame `df` with appropriate data):
# dashboard = create_plot(df, x='date', y='value', color_col='category', title='My Dashboard')
# dashboard.show()


# Example usage (assuming you have a pandas DataFrame `df` with appropriate data):
# dashboard = create_plot(df, x='date', y='value', color_col='category', title='My Dashboard')
# dashboard.show()




" ####################################### CONNECT AND GET DATA ######################################### "

def main():
    #search_string = get_user_input()
    conn, cursor = connect_to_local()
    all_trds_df = get_trds(conn, cursor)
    all_trds_list = reversed(all_trds_df['trd_name'].values)
    print(all_trds_df.columns)
    # options = ['TRD0492', 'TRD0482', 'TRD0443', 'TRD0479', 'TRD0497']
    options = all_trds_list
    search_string = get_user_input(options)
    print(f"Selected option: {search_string}")

    # search_string = 'TRD0492'
    if search_string:
        print(f"Search String Entered: {search_string}")

        tables = get_all_tables(conn, cursor)
        # print(tables) # for troubleshooting
        wellplate_tables = get_wellplates(conn, cursor)
        wellplateca_tables = get_wellplate_CAs(conn, cursor)
        # print(wellplateca_tables.columns)
        # devices = get_devices(conn, cursor)

        trds_df = get_trds(conn, cursor)
        # Search for an exact match in the database table
        # search_string = 'TRD0492'

        matching_ids = search_trd_name(trds_df, search_string)
        print("Matching IDs:", matching_ids[0])
        trd_id = matching_ids[0]
        trd_devices = get_trd_devices(conn, cursor, trd_id)
        print(trd_devices.columns)
        # 'id', 'manufacture_date', 'notes', 'device_name', 'shorted', 'tbl_id',
        #'trd_id', 'baseline_version_id', 'route_id', 'jmp_label',
       #'device_thickness', 'electrolyte_thickness', 'leaked', 'other_failure',
       #'other_failure_description', 'cv_filename']
        device_id_list = trd_devices['id'].values
        device_list = trd_devices['device_name'].values
        print('device list: ', device_id_list)


        trd_eccheckins = get_trd_eccheckins(conn, cursor, device_id_list)
        path_list = trd_eccheckins['server_path'].values # WHY WOULD THERE BE NONE VALS HERE?
        path_list = [item for item in path_list if item is not None]
        print('old path list: ', path_list)

        local_paths = get_local_paths(path_list)
        print('local path list: ', local_paths) 

        # Define the column names
        columns = ['Time', 'Voltage (V)', 'Current (mA)', 'Charge (C)', 'Step Number',
           'Programmed Voltage (V)', 'Programmed Current (A)', 'Control Mode',
           'id', 'cycle']
        # Create an empty DataFrame with the specified columns
        final_df = pd.DataFrame(columns=columns)
        for path in local_paths:
            ### NEED TO ADD CATCH IF GOOGLE DRIVE FILES (not just folders) ARE DELETED OR NOT CONNECTED PROPERLY!
            try: 
                file = path.split('/')[-1]
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
        print(final_df)

        # GETTING INITIAL PHOTOS
        try:
            if not local_paths:
                raise FileNotFoundError("Example file not found error.")
            else:
                photo_path_folder = get_initial_photo_path(local_paths)[0]
                # Example usage
                photo_paths = get_all_file_paths(photo_path_folder)
                print(photo_paths)

        except FileNotFoundError as e:
            print(e)
            photo_paths = []
            path = create_blank_image_with_text('No Initial Photo Available', 'no_initial_photo_available.jpg')
            photo_paths.append(path)
            print("Photo unavailable for requested cycle. Created blank image.")

        trd_name = search_string

        plot1_inputs = (final_df, 'Time', 'Current (mA)', 'cycle', 'id', 'Current vs Time')
        plot2_inputs = (trd_eccheckins, 'cycle_number', 'tint_max_current', 'device_id', 'tint_max_current_time', 'Maximum/Nucleation Current vs Cycle')
        dashboard = create_dashboard(plot1_inputs, plot2_inputs, trd_name, trd_devices, photo_paths)
        #Index(['id', 'measurement_date', 'ec_check_in_file', 'device_id',
       #'bleach_time', 'bleach_voltage', 'coulombic_efficiency', 'cycle_number',
       #'tint_time', 'tint_voltage', 'importData', 'bleach_final_current',
       #'bleach_max_current', 'tint_final_current', 'tint_max_current',
       #'tint_charge_a', 'tint_charge_b', 'tint_charge_c', 'tint_charge_d',
       #'charge_in', 'charge_out', 'local_path', 'server_path',
       #'tint_max_current_time'],
        dashboard.show()
    else:
        print("No search string entered. Exiting.")

if __name__ == "__main__":
    main()


