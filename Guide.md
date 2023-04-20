# Guide: 


Bellow is the final refined step-by-step guide for coding programs in python using GPT-4. The guide contains links that will help you jump to the part of the guide you need to visit.

  

Here are the steps to follow when trying to use GPT-4 for writing code in python (it will work for other programming languages as well if we specify we want to code in them): 

  

<a id="step1"></a> **Step 1 – Preparation**

+	**a**.<a id="step-1a"></a> Open up a new conversation and copy the [**Prompt 1-a**](#prompt-1a). Afterwards, explain to the model in detail what are you trying to code. It will ask you some questions to get a better understanding of what you are trying to accomplish. The model will then create a detailed list of steps we need to go through along with detailed explanations. This list is mostly for us so that we know what we want to do next. If at any point the model doesn’t finish its response – meaning that it has reached its maximum response length – either type the “continue” or tell the model to only code one function at a time or break a code it is coding into more parts (whatever is applicable). 

+	**b**.<a id="step-1b"></a> Pass into the model the [**Prompt 1-b**](#prompt-1b). This prompt will make the model write a python pseudocode that contains explanations of what its parts do, names of variables the different parts of the code use plus their types (GPT-4 can also use other programming language if you change the prompts a little, but the model is best at using python and it is a programming language that is easily readable by humans). This pseudocode is much shorter than the final program. We will use the pseudocode repeatively in the next step as the model’s reference of what variables it should use and what other parts of the code do. This will allow the model to have accurate information about the code. Since the pseudocode will be relatively short, the model will remember it much better than if we always gave it the whole code.  

+	**(optional) c**.<a id="step-1c"></a> Pass the model the [**Prompt 1-c**](#prompt-1c) until it tells you the pseudocode is complete. The goal of this step is to ensure our pseudocode contains as much of the functions and classes as possible since many parts of the code will require few custom functions we need to code and many of them probably aren’t used in the pseudocode. 

  

<a id="step2"></a> **Step 2 – Coding** – repeat the steps bellow for each part of the code until it works for all functions, including the main function that will run the code (this coding loop is much easier to do in jupyter notebooks because they allow one to run parts of code separately): 

+ **a**.<a id="step-2a"></a> Open up a new conversation and pass the model the [**Prompt 2-a**](#prompt-2a) and modify it to reflect what part of the code you will work on during this coding cycle. 

+ **b**.<a id="step-2b"></a> Pass the model the pseudocode from **Step 1-b**. and let it generate the code. 

+ **c**.<a id="step-2c"></a> Test the model’s code in the way it suggests (usually a simple test function – this coding loop is much easier to do in jupyter notebooks because they allow one to run parts of code separately). 

+ **d**.<a id="step-2d"></a> If there was no error, pass the printed statements to the model and if it says everything works as it should, end here and begin the cycle anew with the next part of the code. 

+	**e**.<a id="step-2e"></a> If there was an error (either the code threw an error or the model thinks the code is incorrect after reviewing the print statements), pass it the [**Prompt 2-e**](#prompt-2e). 

  
<a id="step3"></a> **Step 3 – Finish**

+	**a**.<a id="step-3a"></a> Once the code works as intended, remove all the unnecessary prints. You can do that manually or just use ChatGPT-3.5 (the less advanced, but much faster version of the model) in following way: pass the model [**Prompt 3-a**](#prompt-3a) and copy the part of the code from which you want to remove the prints (I recommend doing so for each function separately).   

End of this step marks the end of the coding - congratulations, your code is finished. 

    
<br>
<br>
<br>

## The final version of the prompts used by the guide above: 

   

<a id="prompt-1a"></a>**Prompt 1-a**

 Your goal is to code a program in Python. After I explain this goal, you need to ask me at least 10 questions about the program you are about to code, making sure you get all the necessary information. Your questions should cover topics such as data used by the program and any optimizations I might want to include. Continue asking questions until I agree that you have all the necessary information. Once you have gathered enough information, create a detailed list of steps that need to be taken to code the specified program. For each step, provide a deep, detailed explanation of why it is important. If you understand, say "Understood" and proceed by asking me questions about the program.(**Back to** [**Step 1-a.**](#step1))

  

 
<a id="prompt-1b"></a>**Prompt 1-b** 

  

+ **First prompt:**

Your goal is to turn a list of steps for coding a program into a Python pseudocode that follows a specific format provided later. First, I will provide you with the list of steps. Next, I will give you the format or an example of the Python pseudocode you should follow. After receiving both pieces of information, create the pseudocode based on the steps and the format provided. If you understand, say "Understood" and I will proceed by giving you the list of steps. 
  

+ **List of steps:** 

Copy of the list from [**Step 1-a**](#step1).

  

+ **Pseudocode format:**  

The pseudocode should contain the names of all the functions and classes we will need to code and the classes should also contain the functions and methods they will use. In each function, the pseucode needs to write in """comments""" what the function does and what variables the function returns. The pseudocode should include the main function which runs the whole program. Bellow is an example of this format for your reference (please note that variables like var_name means you should provide real names of the variables in your pseudocode and ... means that things like explanation of how the function works or all returned variables should be written there in the actual pseudocode you create): 

  

def fn_name (arg_name1, arg_name2,...): 

    """This function does ... 

    We use this function to ... 

    return var_name1, var_name2, var_name3,... 

    """ 

    pass 

  

 

class Class_name: 

    def __init__ (self, arg_name1,...): 

        """This initializes class Class_name. The purpose of the Class_name is... 

        We use fn_name in initialization to ... 

        """ 

    def class_method_name (self, arg_name1,...): 

        """This method does ... 

        We use this method to ... 

        return var_name1, var_name2,... 

  

    ... other class methods 

  

def main(): 

    ... 

  

if __name__ == "__main__": 

    main() 

(**Back to** [**Step 1-b.**](#step-1b)) 

  

<a id="prompt-1c"></a>**Prompt 1-c**

  
Your goal is to check the pseudocode I provide later and ensure that all functions and methods the finished program would use are included in the pseudocode. If any custom functions or methods that need to be coded for the final program are missing, you should add them to the pseudocode. Keep in mind that the new parts of the pseudocode you create must follow the same format as the rest of the pseudocode. I will provide you with the pseudocode in the next prompt. After receiving the pseudocode, review it and make any necessary additions. If you understand, say "Understood" and I will proceed by giving you the pseudocode.(**Back to** [**Step 1-c.**](#step-1c)) 


  

<a id="prompt-2a"></a>**Prompt 2-a**

Your goal is to code an implementation of a part of the pseudocode specified later. The code you create must print every variable it uses right after it is used or declared, and/or returns. The print statement should include the variable name, value, type, and shape (if applicable). Additionally, print the arguments of any function or method you code right at the beginning of the function or method in the same way. I will provide you with the pseudocode in the next prompt. After that, I will ask which part of the code you are working on. Then, you should ask at least four questions about the specified part of the code. Once you receive this information, code the specified part of the pseudocode and create a test to check whether the code is working correctly. Then, I will provide you with the text from the printed statements. After receiving this information, review it to ensure the code works as expected. If you understand, say "Understood" and I will proceed by giving you the pseudocode.(**Back to** [**Step 2-a.**](#step2)) 


  

<a id="prompt-2e"></a>**Prompt 2-e** 

Your goal is to fix an error in the code. You should think about all the things that could have gone wrong and the information you would need to correct the error. I will provide you with the text from print statements that resolved before the error and the error message. After receiving this information, write at least three different things you think might have caused the error and rank the likelihood of that being the case, with 0 being 0% chance and 10 being 100% chance. Next, ask at least five questions that will give you the information needed to resolve the error, focusing more on the problems with higher likelihoods. After gathering the necessary information, guide the user in solving the error, providing instructions as if they were a 5-year-old child. If you understand, say "Understood" and I will proceed by giving you the print statements and error message. Continue with this process until the error is resolved.(**Back to** [**Step 2-e.**](#step-2e)) 

  

  

<a id="prompt-3a"></a>**Prompt 3-a** 

  

Your goal is to remove all unnecessary print statements from the code snippets I will provide you with and write the updated code snippets. These print statements were initially used to make troubleshooting of the code easier, but they are now redundant. I will provide you with a code snippet to rewrite. After receiving the code snippet, rewrite it in a way that leaves the program unchanged but removes unnecessary print statements. If you understand, say "Understood" and I will proceed by providing you with a code snippet. Repeat this process each time I provide you with a new code snippet.(**Back to** [**Step 3.**](#step3))

<br>
<br>

## Possible Issues and Their Solutions

1. The model doesn't wait for me to provide the pseudocode format or it starts coding before I tell it which part of the pseudocode I want to code.

**Solution:** Simply give the model the pseudoprompt or the name of the function you want to code right now in the next prompt.

2. The model doesn't do what I instructed it to do.

**Solution:** If the model simply misinterpreted your request, tell it what it did wrong and why its current response isn't what you had in mind.

3. The model forgets what happened previously in the conversation.
**Solution:** The model ran out of memory. If the function or class to code was so complex and so long that the model couldn't code the function during the course of a single conversation, do this:

+ If the code you couldn't finish is a function, break the function into smaller parts that you will code separately.
+ If the code you couldn't finish is a class, code its parts (methods) separately and if the code to implement is still too long, break the method into smaller parts (like with the functions) that will be coded and tested separately.

4. The model tells me to try the "updated" code, which is exactly the same as before.

**Solution:** Simply tell the model that it didn't change the code or its functionality and provide it with the code it was originally supposed to update for reference.

5. The model uses different names for variables, functions, etc., than it was supposed to use, even though it used them correctly previously in the conversation.

**Solution:** The model can quickly forget parts of a lengthy code. When this happens, tell it that it didn't use the correct name/parts of the code, and give it the full code of the part that causes the issue:
+ If it doesn't use functions correctly, pass it the full code of any function it forgot.
+ If the model uses wrong names for variables, provide it with a code snippet where the variables were initiated/used.

  
