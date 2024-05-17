-------------------
|Cell_tower.csv   |
-------------------
* a user can have multiple locations, where they have more 
than one pick one where they are most often, or order by alphabetical 
order to break a tie

user_id => identifier for a user
lat  => coordinate value obtained from a nearby cell tower
lng  => coodinate value obtained from a nearby cell tower

-------------------
| Subscriber.csv  |
-------------------
* a user can have multiple subscriptions, but a subscription
can only have 1 user

transaction_date   =>when subscription was created
subscription_id    => unique identifier for a subscription
user_id            => unique identifier for a user
subscription_start => date subscription became active
subscription_end   => date subscription was cancelled
subscription_status =>status of subscription (Slowly Changing Dimension Type 1)
provider            =>Vendor where we offer access to a service
service             => the service that subscription relates to    

-------------------
|Transactions.csv |
-------------------
*Shows all billing attempts for a subscription on a particular day. 
A subscription can be billed multiple times in a day. 

transaction_date  => date charge was attempted
subscription_id   => identifier for a subscription
transactionstatus => status of charge attempt
charge in cent    => amount charged in cents

-------------------
|Subscription_Data.csv |
-------------------

subscription_id	  => The ID of the subscription
club_id	          => The ID of the club the user subscribed to
created           => The date-time the subscription entry was created
channel           => The channel via which the user subscribed
subscription_start => The date-time the subscription started
subscription_end   => The date-time the subscription ended (blank if subscription was still active on 2 May 2021)
total_billed       => The total billed, in cents, throughout the length of the subscription
billing_rate       => The billing rate, in rands
billing_cycle      => The billing frequency


