from h2o_wave import Q, ui, app

@app('/')
async def serve(q: Q):
    # Create a page
    page = q.page

    # Header
    page['header'] = ui.header_card(title='ForgeryGuard', subtitle='Document Forgery Detection')

    # Create a form card with a button to show the form
    form_items = [
        ui.button(name='show_form_button', label='Show Form', primary=True)
    ]

    # Add form fields when the button is clicked
    if q.client.show_form:
        form_items.extend([
            ui.textbox(name='name', label='Name'),
            ui.textbox(name='address', label='Address'),
            ui.button(name='submit_button', label='Submit', primary=True)
        ])

    form_card = ui.form_card(box='1 1 2 2', items=form_items)

    page['form_card'] = form_card

    # Handle button click to show the form
    if q.args.show_form_button:
        q.client.show_form = True

    # Handle form submissions
    if q.args.submit_button:
        submitted_data = {
            'name': q.args.name,
            'address': q.args.address
        }
        # Process the submitted data
        # ...

    await q.page.save()

if __name__ == '__main__':
    app().start()
