OPEN_AI_KEY=OPEN_AI_KEY

PATH_ORIGINAL_DATA="data/Articles_2023.json"
PATH_GT_DATA="data/articles_categories_their_matched_companies.csv"
PATH_NOT_PREPROCESSED="data/chosen_articles.txt"
PATH_4O="data/chosen_articles_cleaned_4o.txt"
PATH_REGEX="data/chosen_articles_cleaned_regex.txt"
PATH_HUMAN="data/chosen_articles_cleaned_by_me.txt"

PATH_TO_SAVE_PREPROCESSED="data/"

LIST_OF_REGEX=[
      'Get the free Morning Headlines email for news from our reporters across the world',
      'Sign up to our free Morning Headlines',
      'email Please enter a valid email address Please enter a valid email address SIGN UP',
      'I would like to be emailed about offers, events and updates from The Independent.',
      'Read our privacy notice',
      'Thanks for signing up to the Morning Headlines email',
      '{{ #verifyErrors }}',
      '{{ message }}',
      '{{ /verifyErrors }}',
      '{{ ^verifyErrors }}',
      'Something went wrong. Please try again later',
      r'Advertisement !- - ad:.*?- -> ',
      r'<!- - ad:.*?- ->',
      r'<a href=.*?</a>',
      r'http.*? ',
      'CLICK HERE TO SUBSCRIBE TO FOX NATION',
      'CLICK HERE FOR MORE SPORTS COVERAGE ON FOXNEWS.COM',
      'CLICK HERE TO GET THE FOX NEWS APP',
      'CREATE FREE ACCOUNT',
      'CLICK HERE TO DONATE TO "MAKE CAMO YOUR CAUSE"',
      'subscribe to my free CyberGuy Report Newsletter by heading to Cyberguy.com/Newsletter CLICK HERE'
      
  ]