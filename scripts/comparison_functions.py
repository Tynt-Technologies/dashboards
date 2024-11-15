import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

def comparison_create_dynamic_comparison_dialog(all_baselines_df, all_trds_df, conn, cursor):
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

import pandas as pd
import panel as pn
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import panel as pn

def remove_outliers_iqr(df, column):
    """Remove outliers from a DataFrame using the IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    # Define bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Remove outliers
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def comparison_create_static_cycling_jmp_panel2(ec_optics_df_1, ec_optics_df_2, selected_name_1, selected_name_2):
    # Define y-variables for plotting
    y_variables = ['coulombic_efficiency', 'tint_max_current', 'tint_max_current_time', 
                   'delta_initial_final_percentage', 'final_percentage', 'initial_percentage',
                   'min_percentage', 'tint_time_eighty_vlt', 'tint_thirty_five_time',
                   'min_VLT_coloration_efficiency', 'min_VLT_psuedo_coloration_efficiency', 
                   'charge_in', 'a_initial', 'b_initial', 'tint_ten_b', 'tint_ten_a']


    # Filter both dataframes for the Ambient route
    ec_optics_df_1 = ec_optics_df_1[ec_optics_df_1['route_name'] == 'Ambient']
    ec_optics_df_2 = ec_optics_df_2[ec_optics_df_2['route_name'] == 'Ambient']

    # Create image panes
    image_panes = []

    # Generate comparison line plots for all relevant variables
    for variable in set(y_variables):
        # Check and average the data for the first dataframe
        if variable in ec_optics_df_1.columns and not ec_optics_df_1[variable].isna().all():
            filtered_df_1 = remove_outliers_iqr(ec_optics_df_1, variable)
            avg_df_1 = filtered_df_1.groupby('cycle_number')[variable].mean().reset_index()
            avg_df_1['Dataset'] = selected_name_1
        else:
            avg_df_1 = pd.DataFrame(columns=['cycle_number', variable, 'Dataset'])

        # Check and average the data for the second dataframe
        if variable in ec_optics_df_2.columns and not ec_optics_df_2[variable].isna().all():
            filtered_df_2 = remove_outliers_iqr(ec_optics_df_2, variable)
            avg_df_2 = filtered_df_2.groupby('cycle_number')[variable].mean().reset_index()
            avg_df_2['Dataset'] = selected_name_2
        else:
            avg_df_2 = pd.DataFrame(columns=['cycle_number', variable, 'Dataset'])

        # Combine the two dataframes for comparison
        combined_df = pd.concat([avg_df_1, avg_df_2], ignore_index=True)

        # Create the plot
        plt.figure(figsize=(12, 6))  # Increased size for taller and wider plots

        if not avg_df_1.empty or not avg_df_2.empty:
            # Plotting using Seaborn
            sns.lineplot(data=combined_df, x='cycle_number', y=variable, hue='Dataset', marker='o')
            plt.title(f'Comparison of {variable.replace("_", " ").title()}')
            plt.xlabel('Cycle')
            plt.ylabel(variable.replace('_', ' ').title())

            # Set y-axis limits based on both datasets
            if not avg_df_1.empty and not avg_df_2.empty:
                min_y = min(avg_df_1[variable].min(), avg_df_2[variable].min())
                max_y = max(avg_df_1[variable].max(), avg_df_2[variable].max())
                plt.ylim(min_y - 0.1 * (max_y - min_y), max_y + 0.1 * (max_y - min_y))
            elif not avg_df_1.empty:
                min_y = avg_df_1[variable].min()
                max_y = avg_df_1[variable].max()
                plt.ylim(min_y - 0.1 * (max_y - min_y), max_y + 0.1 * (max_y - min_y))
            elif not avg_df_2.empty:
                min_y = avg_df_2[variable].min()
                max_y = avg_df_2[variable].max()
                plt.ylim(min_y - 0.1 * (max_y - min_y), max_y + 0.1 * (max_y - min_y))
        else:
            plt.title(f'Comparison of {variable.replace("_", " ").title()}')
            plt.xlabel('Cycle')
            plt.ylabel(variable.replace('_', ' ').title())
            plt.text(0.5, 0.5, 'No Data Available for Both Datasets', horizontalalignment='center', verticalalignment='center', 
                     transform=plt.gca().transAxes, fontsize=15, color='red')
            plt.xlim(0, 1)  # Set x limits for empty plot
            plt.ylim(0, 1)  # Set y limits for empty plot

        plt.legend()
        plt.grid()

        line_plot_filename = f'{variable}_comparison_plot.png'
        plt.savefig(line_plot_filename)
        plt.close()
        image_panes.append(pn.pane.PNG(line_plot_filename, width=600, height=400))

    # Handle the case where there are no plots generated
    if not image_panes:
        return pn.Column('### No Data Available')
    
    # Create a Panel layout to display plots for the Ambient route
    return pn.GridBox(*image_panes, ncols=2, sizing_mode='stretch_width')


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import panel as pn


def remove_outliers_iqr(df, column):
    """Remove outliers from a DataFrame using the IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    # Define bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Remove outliers
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import panel as pn

import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import panel as pn

def remove_outliers_iqr(df, column):
    """Remove outliers from a DataFrame using the IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def comparison_noncycling_plots(checkin_df_dict_1, baseline_devices_1, checkin_df_dict_2, baseline_devices_2, selected_name_1, selected_name_2):
    plots = {}
    
    # Loop through the data sources for both device sets
    for idx, (checkin_df_dict, baseline_devices, label) in enumerate([
        (checkin_df_dict_1, baseline_devices_1, selected_name_1),
        (checkin_df_dict_2, baseline_devices_2, selected_name_2)
    ]):
        
        # Iterate over each type of check-in DataFrame
        for name, df in checkin_df_dict.items():
            baseline_devices = baseline_devices.rename(columns={'id': 'device_id'})
            df = pd.merge(baseline_devices, df, on='device_id')

            # Filtering for the Ambient route
            df = df[df['route_name'] == 'Ambient']

            # Check if the DataFrame is empty after filtering
            if df.empty:
                print(f"No data for Ambient route in {name} for {label}.")
                continue
            
            # Define x and y based on check-in type
            y_label_map = {
                'df_bubbleareacheckin': ('check_in_bubblearea', 'Bubble Area'),
                'df_devicethicknesscheckin': ('check_in_bottom_thickness_cm', 'Bottom Thickness'),
                'df_hazecheckin': ('check_in_haze', 'Haze'),
                'df_weightcheckin': ('check_in_weight_g', 'Weight'),
                'df_meshwidthcheckin': ('check_in_width_um', 'Mesh Width'),
                'df_internalresistancecheckin': ('check_in_internal_resistance', 'Internal Resistance')
            }

            if name not in y_label_map:
                continue
            
            y_column, y_label = y_label_map[name]

            # Create or update the plot
            if name not in plots:
                fig, ax = plt.subplots(figsize=(12, 6))  # Increased size for taller and wider plots
                plots[name] = {'fig': fig, 'ax': ax}
            else:
                fig, ax = plots[name]['fig'], plots[name]['ax']
            
            try:
                # Check if y_column exists in the DataFrame
                if y_column in df.columns:
                    x = df['check_in_age']
                    y = df[y_column]

                    # Calculate average value for each cycle and plot
                    unique_x = sorted(x.unique())
                    avg_y = [y[x == val].mean() for val in unique_x]
                    ax.plot(unique_x, avg_y, label=f"{label} - {y_label}", marker='o', linestyle='-', color=('blue' if idx == 0 else 'red'))
                else:
                    raise KeyError(f"Column {y_column} not found in DataFrame.")
            except KeyError as e:
                print(f"Error: {e}")
                print(f"Available columns: {df.columns.tolist()}")
                ax.text(0.5, 0.5, f'No Data Available for {label}', horizontalalignment='center', verticalalignment='center', 
                        transform=ax.transAxes, fontsize=15, color='red')
                ax.set_xlim(0, 1)  # Set x limits for empty plot
                ax.set_ylim(0, 1)  # Set y limits for empty plot

            # Set up labels, titles, and legend for the final plot
            ax.set_xlabel('Check In Age')
            ax.set_ylabel(y_label)
            ax.set_title(f"{y_label} Check-In Comparison")

            # Adjust y-axis limits based on the range of both datasets
            all_y = [df[y_column].dropna().tolist() for d in checkin_df_dict.values() if y_column in d.columns]
            all_y_flat = [item for sublist in all_y for item in sublist]  # Flatten the list

            if all_y_flat:
                y_min = min(all_y_flat)
                y_max = max(all_y_flat)
                ax.set_ylim(y_min - 0.15 * (y_max - y_min), y_max + 0.15 * (y_max - y_min))

            ax.legend(title="Data Source")
    
    # Save each plot figure
    plot_filenames = {}
    for name, content in plots.items():
        fig = content['fig']
        plot_filename = f"{name}_comparison.png"
        fig.savefig(plot_filename, bbox_inches='tight')
        plt.close(fig)  # Free memory after saving
        plot_filenames[name] = plot_filename
    
    return plot_filenames


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import panel as pn

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import panel as pn

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import panel as pn

def remove_outliers_iqr(df, column):
    """Remove outliers from a DataFrame using the IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def comparison_create_static_noncycling_plot_dictionary_from_df(checkin_df_dict_1, baseline_devices_1, checkin_df_dict_2, baseline_devices_2, selected_name_1, selected_name_2):
    image_panes = []
    
    # Fixed route name
    route = 'Ambient'

    # Define y_label_map for check-in types
    y_label_map = {
        'df_bubbleareacheckin': ('check_in_bubblearea', 'Bubble Area'),
        'df_devicethicknesscheckin': ('check_in_bottom_thickness_cm', 'Bottom Thickness'),
        'df_hazecheckin': ('check_in_haze', 'Haze'),
        'df_weightcheckin': ('check_in_weight_g', 'Weight'),
        'df_meshwidthcheckin': ('check_in_width_um', 'Mesh Width'),
        'df_internalresistancecheckin': ('check_in_internal_resistance', 'Internal Resistance')
    }

    # Create a plot for each measurement type
    for name, (y_column, y_label) in y_label_map.items():
        plt.figure(figsize=(12, 6))

        # Process first dataset
        df_1 = pd.merge(baseline_devices_1.rename(columns={'id': 'device_id'}), checkin_df_dict_1[name], on='device_id')
        df_1 = df_1[df_1['route_name'] == route]

        # Check if y_column exists in the DataFrame and plot if valid
        if y_column in df_1.columns and not df_1[y_column].isnull().all():
            unique_x = sorted(df_1['check_in_age'].unique())
            avg_y_1 = [df_1[y_column][df_1['check_in_age'] == val].mean() for val in unique_x]
            avg_y_1 = remove_outliers_iqr(df_1, y_column)  # Remove outliers

            sns.lineplot(x=unique_x, y=avg_y_1, label=f"{selected_name_1} - {y_label}", marker='o', color='blue')

        # Process second dataset
        df_2 = pd.merge(baseline_devices_2.rename(columns={'id': 'device_id'}), checkin_df_dict_2[name], on='device_id')
        df_2 = df_2[df_2['route_name'] == route]

        # Check if y_column exists in the DataFrame and plot if valid
        if y_column in df_2.columns and not df_2[y_column].isnull().all():
            unique_x = sorted(df_2['check_in_age'].unique())
            avg_y_2 = [df_2[y_column][df_2['check_in_age'] == val].mean() for val in unique_x]
            avg_y_2 = remove_outliers_iqr(df_2, y_column)  # Remove outliers

            sns.lineplot(x=unique_x, y=avg_y_2, label=f"{selected_name_2} - {y_label}", marker='o', color='red')

        # Set up labels and titles
        plt.xlabel('Check In Age')
        plt.ylabel(y_label)
        plt.title(f"{y_label} Check-In Comparison for Route: {route}")
        plt.legend(title="Data Source")
        plt.grid()

        # Save the plot
        line_plot_filename = f'{y_label.replace(" ", "_")}_comparison_plot_{route}.png'
        plt.savefig(line_plot_filename)
        plt.close()  # Free memory after saving

        # Append the image pane for Panel layout
        image_panes.append(pn.pane.PNG(line_plot_filename, width=600, height=400))

    # Create a Panel layout to display all plots
    if image_panes:
        cycle_jmp_layout = pn.GridBox(*image_panes, ncols=2, sizing_mode='stretch_width')
    else:
        print('NO DATA WARNING: No JMP Data for Static Plots')
        cycle_jmp_layout = pn.Column('### No JMP Data to Display')

    return cycle_jmp_layout





import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import panel as pn

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import panel as pn

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import panel as pn



## OLD BASICS 
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

def get_trd_opticscheckins(conn, cursor, device_id_list):
    if conn and cursor:
        device_id_list = [int(id) for id in device_id_list]
        print(device_id_list)
        # Convert the list of IDs to a format suitable for SQL IN clause
        format_strings = ','.join(['%s'] * len(device_id_list))
        print(format_strings)
        sql_query = f'''
            SELECT * 
            FROM tyntdatabase_opticscheckin
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
