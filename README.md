# A/B Testing
Dataset: https://www.kaggle.com/datasets/osuolaleemmanuel/ad-ab-testing/data

Metadata:

Columns Description

  - auction_id: the unique id of the online user who has been presented the BIO. In standard terminologies this is called an impression id. The user may see the BIO questionnaire but choose not to respond. In that case both the yes and no columns are zero.

  - experiment: which group the user belongs to - control or exposed.
      control: users who have been shown a dummy ad
      exposed: users who have been shown a creative, an online interactive ad, with the SmartAd brand. 

  - date: the date in YYYY-MM-DD format

  - hour: the hour of the day in HH format.

  - device_make: the name of the type of device the user has e.g. Samsung

  - platform_os: the id of the OS the user has.

  - browser: the name of the browser the user uses to see the BIO questionnaire.

  - yes: 1 if the user chooses the “Yes” radio button for the BIO questionnaire.

  - no: 1 if the user chooses the “No” radio button for the BIO questionnaire.
