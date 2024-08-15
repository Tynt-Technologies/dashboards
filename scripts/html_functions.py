# TEST CHANGE :)
def all_devices_table(trd_devices_df):
        # HTML Table with inline styling
    table_html = """
    <table style="width: 100%; border-collapse: collapse; margin: 20px 0; font-family:BlinkMacSystemFont; font-size: 12px; color: #FFFFFF;">
        <thead>
            <tr style="background-color: #00564a;">
                <th style="padding: 10px; text-align: left; border: 1px solid #ddd; font-weight: bold;">Device ID</th>
                <th style="padding: 10px; text-align: left; border: 1px solid #ddd; font-weight: bold;">Device Shorted?</th>
                <th style="padding: 10px; text-align: left; border: 1px solid #ddd; font-weight: bold;">Device Route</th>
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
            <td style="padding: 10px; text-align: left; border: 1px solid #ddd;">{trd_devices_df['route_name'].iloc[row]}</td>
            <td style="padding: 10px; text-align: left; border: 1px solid #ddd;">{trd_devices_df['notes'].iloc[row]}</td>
        </tr>
        """
    table_html += """
        </tbody>
    </table>
    """
    return table_html

def generate_devices_report(warmups, o_checks, p_checks, arbin, keyence):
    # HTML Structure with inline styling
    report_html = """
    <html>
    <head>
        <style>
            body {
                font-family: BlinkMacSystemFont, sans-serif;
                font-size: 12px;
                color: #FFFFFF;
                background-color: #333333;
                margin: 20px;
            }
            .section {
                margin-bottom: 20px;
            }
            .header {
                background-color: #d3d3d3; /* Light grey background */
                color: #00564a; /* Dark green font */
                padding: 10px;
                text-align: left;
                border: 1px solid #ddd;
                font-weight: bold;
            }
            .list-item {
                padding-left: 20px;
                color: #000000; /* Black font for bullet points */
            }
        </style>
    </head>
    <body>
    """
    
    # List of headers and corresponding lists
    sections = [
        ('Warmups Found', warmups),
        ('EC/Optics Checkins Found', o_checks),
        ('Photo Checkins Found', p_checks),
        ('Arbin Runs Found', arbin),
        ('Keyence Images Found', keyence)
    ]

    for header, items in sections:
        report_html += f"""
        <div class="section">
            <div class="header">{header}</div>
            <ul>
        """
        for item in items:
            report_html += f'<li class="list-item">{item}</li>\n'
        report_html += """
            </ul>
        </div>
        """
    
    report_html += """
    </body>
    </html>
    """
    
    return report_html
