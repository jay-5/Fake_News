import pandas as pd 

# Reading an inputted csv file with "fake news"
fake = pd.read_csv("Fake.csv")

# Reading an inputted csv file with "true news"
true = pd.read_csv("True.csv")

# setting entire column to 0. When the dataframes are concatenated, this indicates which are real and which are fake
# In this case, the true articles equal to 0
true["fake_news"] = 0 

# In this case, the fake articles are equal to 1
fake["false_news"] = 1 

# we will have to standardize this column to match the other one so that our classifier is actually effective
just_text = true["text"]

# do this with regular expressions. This will extract the text that comes after the hyphen that follows Reuters
just_text = just_text.str.extractall(r"^.* - (?P<text>.*)")

just_text = just_text.droplevel(1)

true = true.assign(text=just_text["text"]) # switcharoo on the columns

#combining fake and true files
df = pd.concat([fake, true], axis = 0)

# These columns are unecessary for the model
df = df.drop(["subject", "date", "title"], axis = 1) 

# Removes the the null records
df = df.dropna(axis = 0)

# save the cleaned csv for other modeling
clean_text = df.to_csv("cleaned_news.csv", index = False) # prevents having two indices