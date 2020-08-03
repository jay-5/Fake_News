import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


df = pd.read_csv("cleaned_news.csv")

# split the data
dependent = "fake_news" # the dependent variable, text is the independent variable here
X = df.drop([dependent], axis = 1) # drop from our X array because this is the text data that gets trained
y = df[dependent]

# Split data into test and train on 75% of the data, test on the rest
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25)

count_vect = CountVectorizer(max_features = 5000)
X_train_counts = count_vect.fit_transform(X_train["text"])  
X_test = count_vect.transform(X_test["text"])


# fit the training dataset on the NB classifier
Naive = MultinomialNB()
Naive.fit(X_train_counts, y_train)


# predict the labels on validation dataset
predictions_NB = Naive.predict(X_test)

# Use accuracy_score function to get the accuracy
print("Accuracy Score:",accuracy_score(predictions_NB, y_test) * 100)

#test

# link: https://www.nytimes.com/2019/03/24/technology/venture-capitalists-ipo-pinterest.html
nyt = [""" SAN FRANCISCO — For Rick Heitzmann, a venture capitalist at FirstMark Capital, the moment is now.
Two companies that FirstMark invested in years ago, Pinterest and Postmates, have filed paperwork to go public. Pinterest unveiled an offering prospectus on Friday. A third company, Airbnb, is expected to do the same within the next year. All are likely to be embraced by Wall Street, generating huge profits for FirstMark.
Which means it’s time for Mr. Heitzmann, whose New York investment firm has kept a relatively low profile, to consider hiring a public relations agency. The flood of initial public offerings is an occasion for FirstMark to “tell our story,” he said. “We have been part of some of the more successful start-ups in the last several years and have been instrumental to their growth, often as the first investor. That’s a good story to tell.”
Across tech land, investors who helped nurture “unicorns,” the start-ups valued at more than $1 billion, are now getting ready to embody a different animal metaphor: the peacock.
That’s because the companies they bet on years ago — such as Uber, Lyft, Pinterest, Slack, Postmates and Peloton — are preparing to list on the stock market, which would create a bonanza of returns, the likes of which Silicon Valley hasn’t seen for years.
For venture capitalists, that means it is time to stop walking a tightrope of self-promotion. For years, many did not want to look as if they were taking credit for the success of the start-ups they backed. And they did not want to make it seem that the truckloads of money they stood to make were more important than their start-ups’ high-minded missions.
So they sent fawning congratulatory tweets, posted generic blog posts to Medium and name-dropped their deals on one of a dozen interchangeable V.C. podcasts. They “thought led” and became “thinkfluencers.”
Now that they’re ready to shed the false modesty, some investors are calling in the pros. Ann Miura-Ko, a partner at the venture capital firm Floodgate, recently worked with the public relations firm OutCast to spotlight her “special relationship” with the ride-hailing company Lyft. Ms. Miura-Ko invested in Lyft in 2010, when it was called Zimride, and her firm owns about 1.5 million shares in the company. That stake could be worth as much as $102 million when Lyft goes public, which could happen this week.
Ms. Miura-Ko said Lyft’s I.P.O. would validate Floodgate’s strategy of investing in companies when they are just starting, before the bigger venture firms get involved. “We’ve had great exits along the way, but this is the real proof point,” she said. “We think it’s important to Floodgate’s story.” When asked about hiring a public relations firm, Ms. Miura-Ko deflected. Later, she said, “Every person has their own way of developing a megaphone.”
In a follow-up statement, she added, “I’m a teacher and an investor, I have a Ph.D. in math modeling, and I think like an engineer, so thinking about myself and my business from a branding and marketing perspective doesn’t come naturally to me, but these are important factors in the success of all kinds of businesses all over the world, which is why I engaged OutCast to help me think through and formalize both the Floodgate brand and my professional brand.”
OutCast declined to comment. In Silicon Valley, some public relations firms said they had noticed more queries from venture capitalists in the last month. Mike Moeller, founder of Aircover Communications, a public relations firm, said it had been busier than normal with cold calls from venture firms that were trying to stand out.
“Some firms are realizing, ‘Huh, maybe awareness doesn’t just magically grow on trees,’” he said.“They’re realizing they need to be part of this conversation,” added John Kuch, a public relations executive at Moxie Communications Group. With so much money in tech, “folks who have been conservative are getting out there even more.”
For years, venture capitalists on Sand Hill Road, the epicenter of the venture industry in Menlo Park, Calif., were a more buttoned-up lot. Few used public relations firms. Many investors looked askance at talking to the press.
But that changed as venture capital increasingly became a reputation game. Venture firms such as Andreessen Horowitz, which was co-founded by the Netscape wunderkind Marc Andreessen, hired their own public relations staff.
Some venture firms built out entire media divisions. First Round Capital, for example, created a digital magazine, First Round Review, which publishes entrepreneurial profiles, advice and lessons. Other investors have found time between start-up pitches and board meetings to publish books, deliver TED talks and amass huge followings on Twitter.
Still, nothing crowns them as much as being associated with a unicorn company that goes public. When Facebook went public in 2012, venture capitalists who had invested in the social network solidified their reputations as rainmakers — and also were set for life. They included Jim Breyer of Accel Partners, David Sze of Greylock Partners and Peter Thiel of Founders Fund.
“A $9 Billion Jackpot for Facebook Investor,” one headline screamed about Mr. Breyer in 2012, marveling at how he had turned a $12.7 million investment in the social network into a pile of gold.
That kind of name recognition gives investors an edge over competition in new deals. Promotion is part of the game, said Lisa Wu, a partner at Norwest Venture Partners. “If you don’t do it, you miss out,” she said.
Trevor O’Brien, who is working on a start-up in San Francisco, said entrepreneurs also looked at whether a venture capitalist had invested in any unicorns.
“V.C.s showcasing logos from big investments does matter in terms of signaling credibility and experience,” he said, adding that while it is not the only factor, “it checks a box.”
Alongside Ms. Miura-Ko and Mr. Heitzmann, this wave of I.P.O.s will likely raise the profiles of Jeff Jordan, a partner at Andreessen Horowitz who backed Airbnb and Pinterest, and Alfred Lin, a venture capitalist at Sequoia Capital who invested in Airbnb.
Mr. Heitzmann said FirstMark was not embarking on a promotion effort because of trouble raising new funds or getting into good deals. Instead, founders of younger companies that FirstMark has invested in have suggested he try to raise the firm’s profile for their sake.
“Portfolio companies have said, ‘If you guys had a better brand, it would help me recruit,’” he said. Ms. Miura-Ko said she didn’t do any personal branding when Floodgate started in 2010 because there wasn’t much competition to invest in the very young companies that the firm was targeting.
“You didn’t have to think about it,” she said. But now it’s a lot noisier. Ms. Miura-Ko was recently told by students at Stanford University, where she teaches classes about start-ups, that to stand out, she should create her own channel on Discord, a chat app for gamers. She was amused by the suggestion but didn’t follow it, she said. Why? Because it didn’t feel authentic to her personal brand.
""" ]

nyt_vec = count_vect.transform(nyt)
predict_nyt = Naive.predict(nyt_vec)

def classifier(text):
    Naive = MultinomialNB()
    Naive.fit(X_train_counts, y_train)
    
    # n.b: you may need to wrap the argument in brackets to make it a vector if you passed in a string
    word_vec = count_vect.transform(text) 
    
    predict = Naive.predict(word_vec)
    return "Fake News Story" if predict[0] else "Real News Story"

classifier(nyt)
