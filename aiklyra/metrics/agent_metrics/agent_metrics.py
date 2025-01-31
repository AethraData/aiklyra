from pydantic import BaseModel
from openai import OpenAI
import typing as t
import os 


class LLMClient:
    judge_model = None 

    @staticmethod
    def set_model(model_instance):
        LLMClient.judge_model = model_instance

class FaithfulnessSchema(BaseModel):
    score: float 
    reason : str

class ScoreSchema(BaseModel):
    score: float 
    
    
    

class AgentMetrics:
    """
    A class to analyze agent metrics based on user input and reference topics.
    Attributes:
    -----------
    user_input : List[dict]
        A list of dictionaries containing user inputs.
    reference_topics : List[str]
        A list of reference topics.
    system_prompt : str
        The system prompt loaded based on the task name.
    """
    def __init__(self,user_input:t.List[t.Union[str,dict]],reference_topics:t.List[str],task_name:t.Literal["agent_goal_accuracy","topic_adherence","faithfulness"],llm_answer:t.Optional[str]=None,model_instance:OpenAI=None):
        """
        Initializes the AgentAnalyser with user input, reference topics, and a task name.

        Args:
            user_input (List[dict]): A list of dictionaries containing user input data.
            reference_topics (List[str]): A list of reference topics.
            task_name (str): The name of the task to load the system prompt for.
        """
        self.prompt=self.construct_prompt(user_input=user_input,reference_topics=reference_topics,llm_answer=llm_answer)
        self.task_name=task_name
        LLMClient.set_model(model_instance)
        self.system_prompt = self.__load_system_prompt(task_name)

    @classmethod
    def construct_prompt(self,**kwargs):
        """
        Constructs a prompt based on the user input and reference topics.
        Returns:
            str: The constructed prompt.
        """
        if kwargs['llm_answer'] is not None:
            return f"User Questions : \n {kwargs['user_input']} \n LLM Answer : {kwargs['llm_answer']} \n Reference Topics : \n {kwargs['reference_topics']} \n"
        
        return f"User Questions : \n {kwargs['user_input']} \n Reference Topics : \n {kwargs['reference_topics']} \n"
    @classmethod
    def __load_system_prompt(self,task_name:str)->str:
        """
        Load the system prompt for a given task.

        Args:
            task_name (str): The name of the task for which to load the system prompt.

        Returns:
            str: The content of the system prompt file as a string.

        Raises:
            FileNotFoundError: If the specified prompt file does not exist.
            IOError: If there is an error reading the prompt file.
        """
        try : 
            base_path = os.path.dirname(os.path.dirname(__file__))
            prompt_path = os.path.join(base_path,"prompts",f"{task_name}.txt")

            with open(prompt_path, "r") as f:
                return f.read()
        except FileNotFoundError or IOError as e:
            raise f"Error : {e} "

    def _get_score(self):
        """
        Generates a response from the judge_model based on user input and reference topics.
        This method constructs a message payload with the system prompt, user input, and reference topics,
        and sends it to the judge_model's chat completion API to generate a response.
        Returns:
            str: The content of the response message from the judge_model.
        Raises:
            ValueError: If an exception occurs during the API call, the exception is caught and re-raised as a ValueError.
        """
        if LLMClient.judge_model is None : 
            return "Error : Model not set"

        try : 
            messages=[{"role":"system","content":self.system_prompt},{"role":"user","content":self.prompt}]
            result = LLMClient.judge_model.beta.chat.completions.parse(
                model="gpt-4o",
                messages=messages,
                response_format= FaithfulnessSchema if self.task_name=="faithfulness" else ScoreSchema,
            )
            
            return result.choices[0].message.content
        except Exception as e:
            raise ValueError(e)