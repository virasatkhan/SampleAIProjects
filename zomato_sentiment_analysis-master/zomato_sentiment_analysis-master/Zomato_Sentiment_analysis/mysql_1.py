import MySQLdb
from zomato_testing_sentiment_classifier import sentiment_review
from zomato import Zomato
from mail_building import mail_sending
import json

# Creation of Database and table if alreay created then hash the codes
db = MySQLdb.connect(host="localhost", user = "root",   # your host, usually localhost
                     passwd="ankiosa#084507")# your password
cursor = db.cursor()
sql = 'CREATE DATABASE zomato_restaurant_review_sentiment'
cursor.execute(sql)
sql = 'USE zomato_restaurant_review_sentiment'
cursor.execute(sql)
sql = 'CREATE TABLE restaurant_sentiment_16774318 (review_id VARCHAR(13),user_name VARCHAR(50),user_profile VARCHAR(50), rating VARCHAR(4),review_text VARCHAR(500), timestam VARCHAR(12),review_sent VARCHAR(3))';
cursor.execute(sql)

#Extracting Reviews from particular restaurant
z = Zomato("66ae423d7084dab625ce29bdcdba3b28")
output = z.parse("reviews","res_id=16673760")#16774318") #16673760
output_json = json.loads(output)

#Checking if it exists in MYSQL Database or not
# Deleting is done to remove 3 of the reviews to show how its work
for delete_count in range(3):
    review_id = output_json['user_reviews'][delete_count]['review']['id']
    query = "DELETE FROM restaurant_sentiment_16774318 WHERE review_id = %s"
    cursor.execute(query, (review_id,))
    db.commit()
    
for review_count in range(5):
    review_id = output_json['user_reviews'][review_count]['review']['id']
    review_ids = str(review_id)
    sqlq = "SELECT COUNT(1) FROM restaurant_sentiment_16774318 WHERE review_id = %s" % (review_ids)
    cursor.execute(sqlq)
    count = cursor.fetchone()[0]
    if not count:
        user_name = output_json['user_reviews'][review_count]['review']['user']['name']
        user_profile = output_json['user_reviews'][review_count]['review']['user']['profile_deeplink']
        rating = str(json.loads(output)['user_reviews'][review_count]['review']['rating'])
        review_text = output_json['user_reviews'][review_count]['review']['review_text']
        timestam = str(output_json['user_reviews'][review_count]['review']['timestamp'])
        review_sent = sentiment_review(review_text) #Classifying the review text
        print user_name,rating,review_sent
        print type(review_id),type(user_name),type(user_profile),type(rating),type(review_text),type(timestam),type(review_sent)
        sql = "INSERT INTO restaurant_sentiment_16774318 VALUES (%s,%s,%s,%s,%s,%s,%s)"
        info = (review_ids,user_name,user_profile,rating,review_text,timestam,review_sent)
        cursor.execute(sql,info) #Storing into MYSQL Db
        db.commit()
        # If rating<=3 or (review_text is negative and rating<=4) send mail to restaurant
        if(float(rating)<=3 or (review_sent=="neg" and float(rating)<=4)):
            mail_sending(user_name,user_profile,rating,review_text,review_sent)
