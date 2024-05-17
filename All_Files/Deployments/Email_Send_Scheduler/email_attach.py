#!/usr/bin/env python
# coding: utf-8

# In[7]:


import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

fromaddr = "3621029@myuwc.ac.za"
toaddr = "3621029@myuwc.ac.za"

msg = MIMEMultipart()

msg['From'] = fromaddr
msg['To'] = toaddr
msg['Subject'] = "Hello"
body = """Hi

How are you?

Real Python has many great tutorials:
www.realpython.com

Kind Regards
Mncedisi
"""

msg.attach(MIMEText(body, 'plain'))

filename = "fileName"
attachment = open("C:/Users/leemn/Downloads/Fake/politifact_fake.csv", "rb")

part = MIMEBase('application', 'octet-stream')
part.set_payload((attachment).read())
encoders.encode_base64(part)
part.add_header('Content-Disposition', "attachment; filename= %s" % filename)

msg.attach(part)

server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login(fromaddr, 'oudpklnbhwptpjtf')
text = msg.as_string()
server.sendmail(fromaddr, toaddr, text)
server.quit()


# In[ ]:




