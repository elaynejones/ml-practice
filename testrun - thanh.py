import requests
import win32com.client as win32
from datetime import datetime
import re

EMAIL_ACCOUNT = 'sjones9@ups.com' # Your Outlook email address
API_KEY = 'FD939840C890BF92' # Your Access License Number
HEADERS = {'AccessLicenseNumber': API_KEY}
API_VERSION = 'v1'
API_BASE_URL = f'https://onlinetools.ups.com/track/{API_VERSION}/details/'
CA_PATH = r'C:\Users\wkh9tvh\Documents\Code\virtual_env\Lib\site-packages\certifi\cacert.pem'

def track_pic(tracking_number: str) -> dict:
    """
    Returns a dictionary of tracking summary or tracking error
    """
    r = requests.get(f'{API_BASE_URL}{tracking_number}', headers=HEADERS, verify=CA_PATH)
    response = r.json().get('trackResponse').get('shipment')[0].get('package')

    if response == None:
        tracking_summary = {}
    else:
        tracking_summary = response[0].get('activity')[0]
    return tracking_summary

def tracking_details(tracking_number: str, tracking_summary: dict) -> str:
    if tracking_summary == {}:
        mail_body = f"""
            <html>
            <title></title>
            <body>
                <font face='Calibri' size='-0.5'>
                <p>Hello,</p>
                <p>Unfortunately, we are unable to locate this package in our system.</p>
                </font>
            </body>
            </html>
            """
    else:
        location = tracking_summary.get('location').get('address')
        status = tracking_summary.get('status')

        last_city = location.get('city')
        last_state = location.get('stateProvince')
        last_zip = location.get('postalCode')
        last_status = status.get('description')
        last_date = datetime.strptime(tracking_summary.get('date'), '%Y%m%d').date()
        last_time = datetime.strptime(tracking_summary.get('time'), '%H%M%S').time()
    
        if last_city != None:
            mail_body = f"""
                <html>
                <title></title>
                <body>
                    <font face='Calibri' size='-0.5'>
                    <p>Hello,</p>
                    <p>Your request has been received and is being reviewed by our support department. 
                    While we investigate this package, we have set up an email alert with USPS for you
                    to receive updates until the package is delivered.</p>
                    <p>Tracking number:<br>
                    &emsp;{tracking_number}<br>
                    Current package status:<br>
                    &emsp;{last_status}&emsp;{last_date} {last_time}<br>
                    USPS Entry Point:<br>
                    &emsp;{last_city}, {last_state} {last_zip}</p>
                    </font>
                </body>
                </html>
                """
        else:
            mail_body = f"""
                <html>
                <body>
                    <font face='Calibri' size='-0.5'>
                    <p>Hello,</p>
                    <p>Your request has been received and is being reviewed by our support department. 
                    While we investigate this package, we have set up an email alert with USPS for you
                    to receive updates until the package is delivered.</p>
                    <p>Tracking number:<br>
                    &emsp;{tracking_number}<br>
                    Current package status:<br>
                    &emsp;{last_status}</p>
                    </font>
                </body>
                </html>
                """
    return mail_body

def reply_mail(mail_items):
    for mail in mail_items:
        if '_MailItem' in str(type(mail)):
            mail_content = mail.Body
            find_tracking_number = re.search(r'92\d{24}', mail_content)
            reply_all = mail.ReplyAll()
            if find_tracking_number != None:
                tracking_number = find_tracking_number.group(0)
                tracking_result = track_pic(tracking_number)
                mail_body = tracking_details(tracking_number, tracking_result)
                reply_all.HTMLBody = mail_body + reply_all.HTMLBody
                reply_all.Save()
            else:
                continue
        else:
            continue

def main():
    outlook = win32.gencache.EnsureDispatch('Outlook.Application').GetNamespace('MAPI')
    account = outlook.Folders[EMAIL_ACCOUNT]
    folder = account.Folders['Inbox']
    read_folder = folder.Folders['Test'] # Change with the appropriate folder
    reply_mail(read_folder.Items)

if __name__ == '__main__':
    main()