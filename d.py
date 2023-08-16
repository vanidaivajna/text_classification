import pandas as pd
from h2o_wave import Q, ui, app

# Load the Excel data into a Pandas DataFrame
def load_excel_data():
    return pd.read_excel('extracted_data.xlsx')

# Save populated data back to the Excel file
def save_populated_data(data):
    data.to_excel('populated_data.xlsx', index=False)

@app('/')
async def serve(q: Q):
    # Create a page
    page = q.page

    # Header
    page['header'] = ui.header_card(title='Document Extraction', subtitle='Extracted Fields')

    # File upload component
    page['file_upload'] = ui.file_upload(name='document_upload', label='Upload Document', max_size='10MB')

    # Display extracted fields
    if q.client.populated_data is not None:
        extracted_fields = q.client.populated_data.iloc[0]  # Assuming a single row
        page['extracted_name'] = ui.textbox(name='name', label='Name', value=extracted_fields['name'])
        page['extracted_address'] = ui.textbox(name='address', label='Address', value=extracted_fields['address'])

    # Populate button
    page['populate_button'] = ui.button(name='populate_button', label='Populate Extracted Fields', primary=True)

    # Submit button
    page['submit_button'] = ui.button(name='submit_button', label='Submit', primary=True)

    # Handle button clicks
    if q.args.populate_button:
        q.client.populated_data = load_excel_data()  # Load extracted data from the Excel file

    if q.args.submit_button:
        submitted_data = {
            'name': q.args.name,
            'address': q.args.address
        }
        if q.client.populated_data is not None:
            q.client.populated_data.iloc[0] = submitted_data  # Update the populated data
        else:
            q.client.populated_data = pd.DataFrame([submitted_data])  # Create a new DataFrame

        save_populated_data(q.client.populated_data)  # Save populated data back to the Excel file

    await q.page.save()

if __name__ == '__main__':
    app().start()
