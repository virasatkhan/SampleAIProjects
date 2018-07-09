import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText

def mail_sending(user_name,user_profile,mail_rating,mail_review_text,mail_sentiment):
    fromaddr = "ankit.kgpian@gmail.com" #please enter your email address
    toaddr = "ankit.kgpian@gmail.com" #please enter reciever email address
    msg = MIMEMultipart()
    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = "Zomato Negative Review by Walkin Analytics"

    body = mail_body_build(user_name,user_profile,mail_rating,mail_review_text,mail_sentiment)
    msg.attach(MIMEText(body, 'plain'))
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(fromaddr, "your password") #please enter your password
    text = msg.as_string()
    server.sendmail(fromaddr, toaddr, text)
    server.quit()

def mail_body_build(user_name,user_profile,mail_rating,mail_review_text,mail_sentiment):
    mail_rating = float(mail_rating)
    if(mail_rating<=3):
        if(mail_sentiment=="neg"):
            mail_body ="""
Dear Restaurant_abc,
Your Customer """+user_name+""" has given you negative rating and text review is also negative.
Please have a look.

Details :
Customer Name: """+user_name+"""
Rating: """+str(mail_rating)+"""
Text Sentiment: """+mail_sentiment+"""
Review Text: """+mail_review_text+"""
Profile Link: """+user_profile+"""

Thanks,
Walkin Analytics
"""
        else:
            mail_body ="""
Dear Restaurant_abc,
Your Customer """+user_name+""" has given you negative rating.
Please have a look.

Details :
Customer Name: """+user_name+"""
Rating: """+str(mail_rating)+"""
Text Sentiment: """+mail_sentiment+"""
Review Text: """+mail_review_text+"""
Profile Link: """+user_profile+"""

Thanks,
Walkin Analytics
"""
    elif(mail_rating<=4):
            print type(user_profile)
            mail_body ="""
Dear Restaurant_abc,
Your Customer """+user_name+""" has given a text review which is negative.
Please have a look.

Details :
Customer Name: """+user_name+"""
Rating: """+str(mail_rating)+"""
Text Sentiment: """+mail_sentiment+"""
Review Text: """+mail_review_text+"""
Profile Link: """+user_profile+"""

Thanks,
Walkin Analytics
"""
    return mail_body

if __name__=="__main__":
    mail_sending("ankit","ankiosa",3.5,"ok","neg")
