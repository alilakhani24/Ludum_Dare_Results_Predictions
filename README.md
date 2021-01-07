# Ludum_Dare_Results_Predictions

Ludum Dare is a well-known, long-running online game jam. It is essentially a competition in which contestants develop a game in 48 or 72 hours, ideally relating to a particular theme, and receive rankings (1-5 stars) from other contestants. There are two categories: the "compo" (developers work alone, using no premade assets, and have only 48 hours) and the "jam" (contestants may work in teams, using some premade assets, and may take up to 72 hours). Games receive an overall rating as well as ratings for individual categories (fun, theme, innovation, graphics, audio, humour, and mood).

This data set consists of data about all games entered in the last 9 iterations of the Ludum Dare competition, obtained via the public API of the official Ludum Dare site. Each entry is labeled according to its final average score in the "overall" category (from 1 to 5 stars, rounded to the nearest integer, or 0 if the game did not receive enough ratings to officially rank). This is a classification problem with 6 (unbalanced) classes.

For each game, there are several numerical and categorical features available, which are described below. Not all of them will be useful for the purposes of this contest. Additionally, for anyone interested in Natural Language Processing, each game includes a text description which likely contains additional useful data. The vast majority of games also have an associated thumbnail image which may allow for some Computer Vision experimentation.

The training data consists of games entered in LD38 through 45, while the test data comes from LD46. LD46 was the largest Ludum Dare competition yet by a substantial margin, presumably due to a large influx of new participants on account of COVID-19 quarantine; as such, keep in mind that there may be some shift in the distribution between the training and test sets...

## File descriptions
- train.csv - the training set (LD38 - LD45, 21950 entries)
- test.csv - the test set (LD46, 4959 entries)
sampleSubmission.csv - a sample submission file in the correct format
thumbnails - a directory containing thumbnail images (filenames are IDs of the corresponding game)
Data fields
id - A unique integer id.
name - The title of the game.
slug - A url-safe version of the title of the game.
path - Appending this to the base url "https://ldjam.com" gives the url of the game's page on the LD site.
competition-num - An integer describing which competition the game was entered in (38 to 45 for the training data, or 46 for the test data).
category - Either "jam" or "compo".
description - A textual description of the game.
num-comments - The number of comment on the game left by other contestants.
published - The date/time when the game was published.
modified - The date/time when the game was last modified.
version - To be honest, I'm not sure exactly what this encodes, but I figured I might as well leave it in.
feedback-karma - The number of "hearts" (likes) that the developer(s) of this game received for feedback that they left on other games.
ratings-given - The number of ratings given to other games by the developer(s) of this game. (Note: this is a float, for reasons that I am too lazy to explain here.)
ratings-received - The number of ratings this game received from other contestants. (Note: this is also a float.)
links - A list of urls showing where to download and/or play the game (separated by semicolons). May be empty. Be aware that pandas.read_csv may interpret empty strings as NaN.
link-tags - A list of tags describing the content of the links (separated by semicolons). Many of these are from a standardized set of platform descriptions (e.g. "microsoft-windows", "web-html5", and "source code"), but some are natural-language descriptions entered by humans. May be empty.
num-authors - The number of authors/developers who worked on the game.
prev-games - The total combined number of games entered in previous LD competitions (not including competitions prior to LD38) by the authors of this game.
X-average - The game's average score in category X (e.g. "fun-average", "innovation-average" etc.). May be -1, indicating that not enough scores were received, or that the entrant opted out of this category.
X-rank - The game's rank in category X relative to other entrants in the same competition & category. May be -1, indicating that not enough scores were received, or that the entrant opted out of this category.
label - The target to be predicted. Either the average score in the "overall" category, rounded to the nearest integer, or 0 if the game did not receive enough ratings to rank.
