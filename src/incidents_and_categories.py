import numpy as np

def get_category_from_gpt(articles, client):
  """
  Uses OpenAI API to get an answer to prepared prompt for text classification, given articles.
  Returns both answers and confidence of answers.
  Args:
    articles (list): list of articles, which will be classified by GPT model
    client: openai client instance
  """

  answers=["" for _ in range(len(articles))]
  answers_confidence=["" for _ in range(len(articles))]
  for i in range(len(articles)):
    prompt =f"""You are a text classification endpoint, classifying given text into categories:
    human_employee_rights
    diversity_equity_inclusion
    environment
    animal_care
    corporate_transparency
    business_involvement
    political_and_religious_views

    If you are not sure, return other.
    Return only name of the category.

    Texts to classify:

    {articles[i]}
    """
    completion = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=[
        {"role": "user", "content": prompt}
      ],
      logprobs=True,
      top_logprobs=5,
      seed=1,
      temperature=0.0
    )
    answers[i] = completion.choices[0].message.content
    power=completion.choices[0].logprobs.content[0].top_logprobs[0].logprob
    answers_confidence[i] = np.round(np.exp(power)*100, 1)
  return answers, answers_confidence

def get_incident_from_gpt(articles, client):
  """
  Uses OpenAI API to get an answer to prepared prompt for incident recognition, given articles.
  Returns both answers and confidence of answers.

  Args:
    articles (list): list of articles, which will be interpreted by GPT model
    client: openai client instance
  """

  answers=["" for _ in range(len(articles))]
  answers_confidence=["" for _ in range(len(articles))]
  for i in range(len(articles)):
    prompt =f"""You are an incident recognition endpoint, for each given text return incident. Your answer cannot be longer than 3 words.

    Text:

    {articles[i]}
    """
    completion = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=[
        {"role": "user", "content": prompt}
      ],
      logprobs=True,
      top_logprobs=5,
      seed=1,
      temperature=0.0
    )
    answers[i] = completion.choices[0].message.content
    power=completion.choices[0].logprobs.content[0].top_logprobs[0].logprob
    answers_confidence[i] = np.round(np.exp(power)*100, 1)
  return answers, answers_confidence