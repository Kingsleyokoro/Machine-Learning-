users = {
"user1" : "password1",
"user2" : "password2",
"user3" : "password3"
}

def accept_login(users,user,password):
   for key,values in users.items():
       if key==values:
           return True
       else:
           return False
 
            
if accept_login(users, "user1", "password1") :
  print("login successful!")
else :
  print("login failed...")
