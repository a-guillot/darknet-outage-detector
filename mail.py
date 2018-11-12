#!/usr/bin/python3

###############################################################################
# Imports

import smtplib
import configparser
from io import BytesIO
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

try:
    import pyscreenshot
except ModuleNotFoundError:
    pyscreenshot = None

###############################################################################


def send_mail(subject, screenshot=False):
    """ Send a mail with the configuration from mail.conf """

    ###########################################################################

    def extract_config():
        """ Use configparser to extract info from mail.conf """

        config = configparser.ConfigParser()

        config.read('mail.conf')

        try:
            return (
                config.get('mail', 'from'),
                config.get('mail', 'to'),
                config.get('mail', 'user'),
                config.get('mail', 'password'),
                config.get('mail', 'mailserver'),
                config.get('mail', 'port')
            )
        except (configparser.NoSectionError):
            print('Error in or non-existence of the mail.conf file.')
            return [None] * 6

    ###########################################################################

    def create_message(subject, email_from, email_to, image=None):
        """ Create the Mime object """

        def get_image(im):
            """ Return a screenshot to be attached to the mail """
            memf = BytesIO()
            im.save(memf, 'JPEG')
            image = MIMEImage(memf.getvalue())

            return image

        message = MIMEMultipart()
        message['Subject'] = subject
        message['From'] = email_from
        message['To'] = email_to

        if image:
            message.attach(get_image(image))

        return message

    ###########################################################################

    def connect_to_server(mailserver, port, user, pwd):
        """ Connect to the mail server """
        server = smtplib.SMTP(mailserver, port)
        server.ehlo()
        server.starttls()
        server.login(user, pwd)

        return server

    ###########################################################################

    # Get current config
    email_from, email_to, user, pwd, mailserver, port = extract_config()

    if not email_from:
        return -1

    # Server creation
    server = connect_to_server(mailserver, port, user, pwd)

    # Get screenshot as image
    if screenshot and pyscreenshot:
        im = pyscreenshot.grab()
    else:
        im = None

    # Message creation
    message = create_message(subject, email_from, email_to, image=im)

    # Send the actual mail
    server.sendmail(email_from, email_to, message.as_string())

    server.quit()
