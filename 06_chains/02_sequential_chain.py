from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os

load_dotenv()

model = ChatOpenAI(
    model = 'openai/gpt-oss-120b',
    openai_api_key = os.getenv('GROQ_API_KEY'),
    openai_api_base = 'https://api.groq.com/openai/v1'
)

class AudienceAnalysis (BaseModel):
    target_age_group: str = Field(description = 'age range of the audience')
    primary_interest: str = Field(description = 'hobbies, topics they like, topics they care etc.')
    pain_points: list[str] = Field(description = 'common problems they face on the platform')
    platform_habits: str = Field(description = 'How they use the social platform')
    purchase_motivation: list[str] = Field(description = 'What drives them to purchase')

class CampaignStrategy (BaseModel):
    campaign_goal: list[str] = Field(description = 'main objective of the campaign like awareness, engagement, conversation, sales etc.')
    content_tone : str = Field(description = 'Voice or style of content to use according to the age group like GenZ vs Boomers vs Millenial etc')
    key_messages: list[str] = Field(description = 'All core messages to communicate with the product')
    content_angle: str = Field(description = 'Unique approach or hook for the campaign')
    posting_recommendations: str = Field(description = 'Best time/frequency to post for this audience')

class PostCopy (BaseModel):
    opening_attention: str = Field(description = 'Attention grabbing first line to stop the scroll')
    main_body: list[str] = Field(description = 'core content explaining the product, value propositions, enlighting pros, why someone should use that product according to age group')
    emotional_appeal: str = Field(description = 'sentence that connects emotionally with the audience')
    social_proof: str = Field(description = 'credibility element like testimonial, stats, achievements')
    call_to_action: str = Field(description = 'Clear action we want to take from audience')

class Hashtags (BaseModel):
    trending_hashtags: list[str] = Field(description = 'Currently popular hashtags trending for the product')
    niche_hashtags: list[str] = Field(description = 'Specific targeted hashtags for the audience')
    branded_hashtags: list[str] = Field(description = 'Specific product related hashtags or product company related hashtags')
    hashtags_count: int = Field(description = 'recommended number of hashtags for this platform')
    hashtags_strategy: list[str] = Field(description = 'Explaination of why these hashtags are selected or chosen')

class CallToAction (BaseModel): 
    cta_text: str = Field(description = 'The CTA phrase like SHOP NOW, LEARN MORE etc.')
    cta_placement: str = Field(description = 'The CTA placement i.e. where to place it in bio, in middle of post, at the end of post, at button etc.')
    urgency_element: str = Field(description = 'Time-sensitive or scarcity element if applicable')
    link_destination: str = Field(description = 'Where the CTA should re-direct the audience like website/DM/Landing page')
    conversion_optimization: str = Field(description = 'Tips to maximize click-through rate')

class FinalSchema (BaseModel):
    audience_analysis: AudienceAnalysis
    campaign_strategy: CampaignStrategy
    post_copy: PostCopy
    hashtags: Hashtags
    call_to_action: CallToAction

parser1 = JsonOutputParser(pydantic_object = AudienceAnalysis)

parser2 = JsonOutputParser(pydantic_object = CampaignStrategy)

parser3 = JsonOutputParser(pydantic_object = PostCopy)

parser4 = JsonOutputParser(pydantic_object = Hashtags)

parser5 = JsonOutputParser(pydantic_object = CallToAction)

final_parser = JsonOutputParser(pydantic_object = FinalSchema)

template1 = PromptTemplate(
    template = 'Analyse the target audience for this product: {product_name}, {product_description}, {platform} \n {format_instruction}',
    input_variables = ['product_name', 'product_description', 'platform'],
    partial_variables = {'format_instruction': parser1.get_format_instructions()}
)

template2 = PromptTemplate(
    template = 'Based on the audience analysis data {input} \n\n create campaign strategy \n {format_instruction}',
    input_variables = ['input'],
    partial_variables = {'format_instruction': parser2.get_format_instructions()}
)

template3 = PromptTemplate(
    template = 'Using the campaign strategy data {input}\n\n Create the engaging post copy \n {format_instruction}',
    input_variables = ['input'],
    partial_variables = {'format_instruction': parser3.get_format_instructions()}
)

template4 = PromptTemplate(
    template = 'Using the post copy data {input} \n\n Platform: {platform} \n\n Create the appropriate hashtags \n {format_instruction}',
    input_variables = ['input', 'platform'],
    partial_variables = {'format_instruction': parser4.get_format_instructions()}
)

template5 = PromptTemplate(
    template = 'Using this hashtags and campaign data: {input} \n\nCreate a compelling call-to-action \n {format_instruction}',
    input_variables = ['input'],
    partial_variables = {'format_instruction': parser5.get_format_instructions()}
)

chain = template1 | model | parser1 | template2 | model | parser2 | template3 | model | parser3 | template4 | model | parser4 | template5 | model | parser5

result = chain.invoke({
    'product_name' : 'Smartwatch',
    'product_description' : 'A new gen smartwatch which supports voice calling feature and heart beat measurement also identifies Blood Pressure',
    'platform' : 'Instagram'
})

print(result)



