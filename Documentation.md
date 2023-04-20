# Intelligent-Coding-Assistant-Project 

Intelligent Coding Assistant Project: Discover a user-friendly, step-by-step framework that harnesses the power of GPT-4 to facilitate collaborative code-writing in python. Boost your coding efficiency with tailored AI-generated prompts and guidance.  

 

# Introduction: 

 

Since OpenAI made their GPT model available to the public in the form of ChatGPT, people have been using this artificial intelligence for a variety of purposes. One popular application of OpenAI's model is using it for coding. The original ChatGPT utilized GPT-3.5, which was fast and took the world by storm. This model eliminated many issues found in previous chatbots, such as nonsensical responses, and demonstrated an extraordinary ability to provide helpful information to users. Some time ago, OpenAI introduced its improved GPT-4 model to the Plus users of ChatGPT, showcasing the model's enhanced understanding of text and improved memory of conversation context. However, even though its coding capabilities have grown, GPT-4 still struggles to create complex programs without errors. When attempting to fix its own code, the AI often gets caught in an endless correction loop. 

In this project, I decided to test the model's capabilities by having it create a simple transformer model for the Toeba English-German seq2seq dataset. I chose this task because machine learning model code is quite complex and presents many opportunities for errors (e.g., variable types, shapes and dimensions, interaction between classes and functions). I used Python as my programming language of choice. I identified weak points in GPT-4's coding abilities and attempted to create clever prompts and a step-by-step guide to address these weaknesses. I aimed to have ChatGPT create a Transformer using these clever prompts and the guide I made. These tools were mostly successful, making the coding process much more comfortable for me. After completing the initial coding, I noted some areas for improvement in my prompts and refined them along with the step-by-step process. Testing the refined prompts resulted in a smoother coding experience. This project also allowed me to deepen my understanding of the actual code behind the components and layers of Transformer models. In the end, I tested the prompts on 4 random edabit challenges ranked as “expert” difficulty and I effortlessly completed them using the refined guide and prompts. 

  


# Initial Approach and Creation of Simple Transformer: 

At the beginning of the project, I wanted to test GPT-4's capabilities to determine whether someone could use it to code a complex program without having programming skills. To that end, I limited my assistance during the coding process. I asked ChatGPT to create a simple transformer model trained on the Toeba English-German dataset. To better familiarize myself with the actual code used by Transformer models, I initially instructed ChatGPT to code the program using basic libraries like numpy. However, later on, I allowed it to use PyTorch because the model couldn't create correct code using only basic libraries. 

I opened two separate conversations. In the first one, ChatGPT generated the preprocessing code, which it did fairly easily. However, I later discovered that it preprocessed the data incorrectly, leading to poor training data and suboptimal model performance. In the second conversation, ChatGPT created the architecture for the network and the training code. The code produced various type and shape errors, and I had to start new conversations as the model couldn't find a solution. After several more conversations, each solving some issues but eventually falling into infinite correction loops, I had the model start from scratch and use PyTorch for the entire architecture code. 

After investing a significant amount of time, I got the model to finish the simple transformer class and its associated classes (different layers of the neural network) and trained the model on my computer. The model performed poorly, partially due to the overuse of teacher forcing—an optimization technique that should be used less frequently as the model improves. It took approximately 11 long conversations to get the model to this point, and the code still wasn't entirely correct (e.g., excessive teacher forcing). Additionally, the transformer lacked optimization techniques and padding masks, making it quite inefficient. 

By continuing to work with GPT-4 on the simple transformer model, I observed the following weak points in the model's ability to code: 

    • Difficulty remembering every instruction if the prompt was too long. 

    • Rapid memory loss of code it was working on, especially as the code length increased. 

    • Trouble finding errors in the full program due to the numerous classes and functions. 

    • Inability to find the correct solution on the first few attempts if only given an error message and the error's location. 

Furthermore, the model was really bad at correcting shape errors. 

However, I also observed several conditions under which the model worked favorably: 

    • Short and concise instructions. 

    • Working on smaller parts of the code. 

    • Immediate testing of smaller code segments. 

    • Printing values and shapes of variables during troubleshooting to help locate errors. 

Having summarized these observations, I decided to create a series of prompts that anyone could use repetitively and effortlessly to code programs, such as the transformer model I was working on in this project.

 

 

# Prompt Creation: 

The goal of this part of the project was to create prompts that anyone could use to code a program, provided that the language model was capable of coding such a program. One of the main challenges in creating prompts was to convey the intentions for the prompts concisely and precisely, as doing so was time-consuming. To address this challenge, I created the concept of "pseudoprompts". “Pseudoprompts” contain model instructions and prompt intentions, enabling quick alterations and fine-tuning of the final prompts. The final pseudoprompts will be provided in the last part of the project. Below are two pseudoprompt examples to demonstrate how they work: 

**Example 1:** 

"explain 0: goal (fix an error that couldn’t be corrected in a few previous tries by the model) 

explain 1: the model should think about all the things that could have gone wrong and about information it would need to correct the error 

ask(x>=5) 0: ask questions that will give the model information that it needs to resolve the error (viz. Explain 1) 

do 0: after ask 0 (if it happened, otherwise right after explain 1), guide the user in solving the error, guide the user as if they would be a 5 year old child 

do 1: repeat steps from explain 1 to do 0 until the error is resolved 

output 0: „Understood.“ + continue with questions/steps to take" 

 

**Example 2:**  

"explain 0: goal (code an implementation of part of pseudocode specified later in input 0) 

explain 1: the code should use a lot of prints to print out values of variables, their shapes,… to make sure we can find any error that might occur quickly 

input 0: Please, provide me with the pseudocode 

input 1: after input 1, What part of code are you working on? 

do 0: after input 1, code the part of pseudocode specified by input 1 

do 1: after do 0, code a test that will test whether the code from do 0 is working correctly 

input 2: after do 1, Please show me the text from the printed statements so that I might review whether everything is working correctly. 

output 0: „Understood.“ + continue with input 0" 

  
<br> 
<br>  
 
 
A new conversation with ChatGPT was initiated, instructing it to act as a **"pseudoprompt translator"** that converts pseudoprompts into actual prompts for the model. The prompt containing the instructions for translating "pseudoprompts" into actual prompts was refined and fine-tuned, resulting in the final version: 

"Hi. From now on, you will act as a „pseudoprompt translator“. I will give you a „pseudoprompt“ and you will turn it into a prompt I can pass to a large language model (like yourself) to achieve what the „pseudoprompt“ specifies. Pseudoprompts are special sets of instructions, that tell you what the prompt you create should contain. There are two types of pseudoprompts: 

  
Pre-translation pseudopromts – they specify the contents of the prompt you want to create and have a format „instruction_type id: instruction“, where instruction type is one of the types explained bellow, id is there to allow a quick post-translation referencing of pseudocode, and instruction tells you what the resulting translated prompt should look like. Each type of pseudoprompt has its own set of ids (so there can be „explain 1“, „explain 2“,... at the same time as „do 1“, „do 2“,... and so on). 

  
Pre-translation pseudomprompt types (in examples bellow, text in square brackets means that the text depends on what was specified in the pseudoprompt): 

- explain: explain the topic specified by the instruction to the model (examples: explain 0: goal (something) -> "Your goal is to [the something specified]"., explain 1: answer must be in [format] -> "Your answer must be in following format: [the format specified].") 

- do: tell the model that it must do what is specified by the instructions (example: do 1: after do 0, create a code -> "After [what was done in do 0], create the code for [the thing specified in previous part of the prompt like explain].") 

- ask(x: optional): tell the model it needs to ask the user x questions about the topic specified to by the instruction – this will give the model necessary information it needs for its task. The variable x is optional and if it wasn’t given, tell the model to ask as much questions about the topic as it needs. (examples - ask (x:optional) 0: questions about [something] -> "You must ask questions about [something] until you have enough information.", ask (x>=3) 1: questions about [something] -> "You must ask at least three questions about [something].)  

- output: tell the model what it needs to output and what it should do afterwards (example - output 1: understood + continue with ask 1→ „If you understand, say ‚Understood.‘ and ask the questions I specified.“) 

- input: tell the the model that the user will provide specific data specified by instruction (example - input 0: provide me with [something], please -> "I will provide you with [the something specified] in the next prompt.") 


Post-translation pseudoprompts – they allow for transformation of the translation you create. Bellow are the types of pseudoprompts. These do not contain id. 

  
Post-translation pseudoprompt types: 

- refine(type id: optional) … make the part of the prompt translated from specified pseudoprompt more concise. If „type id“ argument isn’t given, refine the whole translation 

- alter (type id: optional, why: optional, how: optional) … change the part of the prompt translated from specified pseudoprompt. If „type id“ argument isn’t given, alter the whole translation. If given, „why“ argument tells you why I want that part of the translated prompt changed and „how“ tells you in which way I want to change it. 

  
The translation must always be in second person ("you") and must not address me - it should be written in a way I would address a model with the prompt. The translation is what I will say to the model. Example of translation: 

  
Pseudoprompt: 

explain 0: goal (code program in python) 

explain 1: the must get all the necessary information before coding 

input 0: after explain 0 and explain 1, provide me with the necessary information, please 

ask (x>=3) 0: after input 1, questions about what was provided in input 0 

do 0: after ask 0, code the program 

output 0: 'Understood.' + continue with input 0 

  

Translation: 

"I'd like you to help me code a program in Python. Before we start, I need to explain a couple of things. First, our goal is to code a program in Python. Second, you are only allowed to code once I give you all the necessary information. I will provide you with necessary information in the next prompt. After that, ask me at least three questions about the information provided to ensure you have enough information. After I give you the answers to your questions, code the program. If you understand, say 'Understood.' and ask me to provide you with the necessary information." 

  

Don’t include elements of translation (type ids like „input 0“) in the final translation.  

  

If you understand, say "Understood" and ask for the pseudoprompt to translate." 

  

<br>  
<br> 
<br> 

Below are the translated examples from the pseudoprompts: 

 

**Example 1:**

 "I'd like you to help me fix an error that couldn't be corrected in a few previous attempts. Our goal is to identify the issue and resolve it. To do this, you should consider all the possible things that could have gone wrong and the information you would need to correct the error. 

If you understand, say "Understood." Then, please ask me at least five questions to gather the information needed to resolve the error. After obtaining the necessary information, guide me in solving the error as if I were a 5-year-old child. Continue this process of identifying issues, asking questions, and guiding me until the error is resolved." 

**Example 2:**

 "I'd like you to help me code an implementation of a part of a pseudocode that will be specified later. Our goal is to write code for the specified part and include many print statements to display the values of variables, their shapes, and other relevant information to ensuret that we can quickly identify any errors that may occur. 

If you understand, say "Understood." Then, please ask me for the pseudocode. After receiving the pseudocode, ask me which part of the code we are working on. Once you have this information, code the specified part of the pseudocode and then create a test to verify whether the code is working correctly. Afterward, ask me to provide the text from the printed statements for review to ensure everything is functioning as intended." 

 

 

 

 

# First version of the guide for using GPT-4 as a clever coding assistant: 
 

Follow these steps when using GPT-4 to write code in Python (this process also works for other programming languages by specifying the desired language): 

  

**Step 1 – Preparation** 

+ **a**. Open a new conversation and input **Prompt 1-a**. Provide a detailed explanation of what you aim to code. The model will ask questions to gain a better understanding of the task and create a detailed list of steps with explanations. This list is primarily for your reference. The prompt also includes an instruction to handle cases where the code length exceeds the model's maximum response length, allowing the creation of effectively longer responses. 

+ **b**. Input **Prompt 1-b** to have the model create a Python (or other programming language) pseudocode with explanations, variable names, and types. The shorter pseudocode will be used repeatedly in the next step as a reference, making it easier for the model to remember details compared to providing the entire code. This marks the end of the preparation step. 

  

**Step 2 – Coding**
– repeat these steps for each part of the code: 
+ **a**. Open a new conversation, input **Prompt 2-a**, and modify it to reflect the code part you'll work on during this coding cycle. 

+ **b**. Provide the model with the pseudocode from **Step 1-b** and let it generate the code. 

 (If the model reaches the maximum response length, input the "continue-command" prompt.) 

+ **c**. Test the model's code as suggested. 

+ **d**. If there are no errors and the model confirms that everything works correctly after reviewing the print statements, proceed with the next part of the code. Otherwise, move to **Step 2-e**. 

+ **e**. If an error occurs (either the code throws an error or the model identifies a problem after reviewing the print statements), share the relevant parts of the error message and prints with the model. If needed, input Prompt 2-e and follow the instructions to resolve the error. 

  

**Step 3 – Finish the Code**

+ **a**. Input **Prompt 3-a** and test the entire code. If there are no errors, proceed to the next sub-step. If an error occurs, repeat **Step 2-e**. 

+ **b**. Once the code works as intended, remove all unnecessary print statements. Congratulations, your programming project is complete! 

  
<br>
<br>
I named the prompts after the step in which they are used to make them easier to use (ie. Prompt 2-e is used in Step 2-e). 

Please note that these steps are for writing new code and not for refining, optimizing, or altering existing code. This process assumes that the entire code can be summarized into a pseudocode that is shorter than GPT-4's maximum response length.

<br>

# Results of testing the guide and the prompts: 
 

The prompts mostly worked well. I opened about seven conversations, not due to correction loops but because I followed the guidelines I created. Many of these conversations were resolved with a few prompts. However, I observed some weaknesses in the prompts and in the approach I used. For instance, during the final part of the code (training the model), the code functioned but not as intended. Masks were included, but teacher forcing was not (what I mean is that the model only used teacher forcing), which led to a correction loop when the model tried to make changes outside the guide. The time-consuming nature of redoing the entire code prompted me to refine the prompts and the guide I use. 

  

To address this, I made several modifications to the prompts like changing the ask(x:optional) instruction in pseudoprompts to ask(x>=value), where the value depends on specific prompts, to encourage the model to ask more questions. In the next part of the project, some prompts  and several steps in the guide were altered, and prompts for error-solving were completely redone to provide the model with as much information as possible. Details of these changes will be provided in the next part of the project. 

  


 

 

# Final version of the guide and prompts: 


Here is the final refined step-by-step guide for coding programs in python using GPT-4 (The guide and the prompts are also included in a better-arranged file called **"Guide.md"**, which you can find in the same repository as this file. In the file **"Guide.md"**, each bold text in the description of steps serves as a link to the prompt or step that is referenced in that description.): 

  

Here are the steps to follow when trying to use GPT-4 for writing code in python (it will work for other programming languages as well if we specify we want to code in them): 

  

**Step 1 – Preparation** 

+	**a**. Open up a new conversation and copy the **Prompt 1-a**. Afterwards, explain to the model in detail what are you trying to code. It will ask you some questions to get a better understanding of what you are trying to accomplish. The model will then create a detailed list of steps we need to go through along with detailed explanations. This list is mostly for us so that we know what we want to do next. If at any point the model doesn’t finish its response – meaning that it has reached its maximum response length – either type the “continue” or tell the model to only code one function at a time or break a code it is coding into more parts (whatever is applicable). 

+	**b**. Pass into the model the **Prompt 1-b**. This prompt will make the model write a python pseudocode that contains explanations of what its parts do, names of variables the different parts of the code use plus their types (GPT-4 can also use other programming language if you change the prompts a little, but the model is best at using python and it is a programming language that is easily readable by humans). This pseudocode is much shorter than the final program. We will use the pseudocode repeatively in the next step as the model’s reference of what variables it should use and what other parts of the code do. This will allow the model to have accurate information about the code. Since the pseudocode will be relatively short, the model will remember it much better than if we always gave it the whole code.  

+	**(optional) c**. Pass the model the **Prompt 1-c** until it tells you the pseudocode is complete. The goal of this step is to ensure our pseudocode contains as much of the functions and classes as possible since many parts of the code will require few custom functions we need to code and many of them probably aren’t used in the pseudocode. 

  

**Step 2 – Coding** – repeat the steps bellow for each part of the code until it works for all functions, including the main function that will run the code (this coding loop is much easier to do in jupyter notebooks because they allow one to run parts of code separately): 

+ **a**. Open up a new conversation and pass the model the **Prompt 2-a** and modify it to reflect what part of the code you will work on during this coding cycle. 

+ **b**. Pass the model the pseudocode from **Step 1-b**. and let it generate the code. 

+ **c**. Test the model’s code in the way it suggests (usually a simple test function – this coding loop is much easier to do in jupyter notebooks because they allow one to run parts of code separately). 

+ **d**. If there was no error, pass the printed statements to the model and if it says everything works as it should, end here and begin the cycle anew with the next part of the code. 

+	**e**. If there was an error (either the code threw an error or the model thinks the code is incorrect after reviewing the print statements), pass it the **Prompt 2-e**. 

  

**Step 3 – Finish**

+	**a**. Once the code works as intended, remove all the unnecessary prints. You can do that manually or just use ChatGPT-3.5 (the less advanced, but much faster version of the model) in following way: pass the model **Prompt 3-a** and copy the part of the code from which you want to remove the prints (I recommend doing so for each function separately).   

End of this step marks the end of the coding - congratulations, your code is finished. 

    
<br>
<br>
<br>

## The final version of the prompts used by the guide above (I also include the specific pseudoprompts used to create the prompts): 

   

**Prompt 1-a**

 Your goal is to code a program in Python. After I explain this goal, you need to ask me at least 10 questions about the program you are about to code, making sure you get all the necessary information. Your questions should cover topics such as data used by the program and any optimizations I might want to include. Continue asking questions until I agree that you have all the necessary information. Once you have gathered enough information, create a detailed list of steps that need to be taken to code the specified program. For each step, provide a deep, detailed explanation of why it is important. If you understand, say "Understood" and proceed by asking me questions about the program. 

  

explain 0: goal (coding a program in python)  

ask(x>=10) 0: after explain 0, questions about the program the model is about to code – until it gets all the information it needs and the user agrees that the model has all the necessary pieces of information - the questions should also be about things like data used by the program or if the user wants to use certain optimizations.  

do 0: after ask 0, make a detailed list of steps that need to be taken to code the program specified by the answers to the questions. Include in the list under each step the deep detailed explanations of each step and why it is important.  

output 0: „Understood“ + continue with ask 0 

  

 
**Prompt 1-b** 

  

+ **First prompt:**

Your goal is to turn a list of steps for coding a program into a Python pseudocode that follows a specific format provided later. First, I will provide you with the list of steps. Next, I will give you the format or an example of the Python pseudocode you should follow. After receiving both pieces of information, create the pseudocode based on the steps and the format provided. If you understand, say "Understood" and I will proceed by giving you the list of steps. 

  

explain 0: goal (turn a list of steps of coding into a python pseudoprompt of specific format  – given in input 1) 

input 0: after give me the list of steps, please 

input 1: after input 0, give me the format or example of the python pseudocode, please 

do 0: after input 0 and input 1, create a pseudocode based on information from input 0 and input 1 

output: „Understood.“ + continue with input 0 

  

+ **List of steps:** 

Copy of the list from **step 1-a**

  

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

  

  

**Prompt 1-c**

  
Your goal is to check the pseudocode I provide later and ensure that all functions and methods the finished program would use are included in the pseudocode. If any custom functions or methods that need to be coded for the final program are missing, you should add them to the pseudocode. Keep in mind that the new parts of the pseudocode you create must follow the same format as the rest of the pseudocode. I will provide you with the pseudocode in the next prompt. After receiving the pseudocode, review it and make any necessary additions. If you understand, say "Understood" and I will proceed by giving you the pseudocode. 

  
explain 0: goal(check the pseudocode provided later and if all functions and methods that the finished program would use aren't in the pseudocode, add them to the pseudocode) 

explain 1: the new parts of the pseudocode you create must follow the same format as the rest of the pseudocode 

input 0: provide me with the pseudocode, please 

do 0: after input 0, if some custom functions and methods which need to be coded that will be used by final program aren't in the pseudocode, write them 

ouptut 0: "Understood." + continue with input 0 

  

**Prompt 2-a**

Your goal is to code an implementation of a part of the pseudocode specified later. The code you create must print every variable it uses right after it is used or declared, and/or returns. The print statement should include the variable name, value, type, and shape (if applicable). Additionally, print the arguments of any function or method you code right at the beginning of the function or method in the same way. I will provide you with the pseudocode in the next prompt. After that, I will ask which part of the code you are working on. Then, you should ask at least four questions about the specified part of the code. Once you receive this information, code the specified part of the pseudocode and create a test to check whether the code is working correctly. Then, I will provide you with the text from the printed statements. After receiving this information, review it to ensure the code works as expected. If you understand, say "Understood" and I will proceed by giving you the pseudocode. 

  
explain 0: goal (code an implementation of part of pseudocode specified later) 

explain 1: the code must print every variable it uses (right after it is used/declared) and/or returns (example: print("var_name: ", var_name, ", type: ", type(var_name), "\n") and if the var_name also has a shape attribute, it should be printed after type on the same line as well. Print arguments of any function/method you code right at the beginning of the function/method in the same way. 

input 0: Please, provide me with the pseudocode 

input 1: after input 0, What part of code are you working on? 

ask (x>=4) 0: after input 1, questions about the part of the code specified 

do 0: after ask 0, code the part of pseudocode specified by input 1 

do 1: after do 0, code a test that will test whether the code from do 0 is working correctly 

input 2: after do 1, the text from the printed statements 

do 2: after input 2, review the information provided in input 2 

output 0: „Understood.“ + continue with input 0 

  

**Prompt 2-e** 

Your goal is to fix an error in the code. You should think about all the things that could have gone wrong and the information you would need to correct the error. I will provide you with the text from print statements that resolved before the error and the error message. After receiving this information, write at least three different things you think might have caused the error and rank the likelihood of that being the case, with 0 being 0% chance and 10 being 100% chance. Next, ask at least five questions that will give you the information needed to resolve the error, focusing more on the problems with higher likelihoods. After gathering the necessary information, guide the user in solving the error, providing instructions as if they were a 5-year-old child. If you understand, say "Understood" and I will proceed by giving you the print statements and error message. Continue with this process until the error is resolved. 

  

explain 0: goal (fix an error in code) 

explain 1: the model should think about all the things that could have gone wrong and about information it would need to correct the error 

input 0: the text from print statements that resolved before the error and the error message 

do 0: after input 0, write at least three different things you think might have caused the error and rank the likelihood of that being the case (from 0 to 10 where 0 is 0% chance and 10 is 100% chance)  

ask(x>=5) 0: ask questions that will give the model information that it needs to resolve the error (the higher the likelihood of one of the things that the model thought about in do 0, the more questions should be asked about that specific problem) 

do 1: after after ask 0, guide the user in solving the error, guide the user as if they would be a 5 year old child 

output 0: „Understood.“ + continue with input 0 until the error is resolved 

  

**Prompt 3-a** 

  

Your goal is to remove all unnecessary print statements from the code snippets I will provide you with and write the updated code snippets. These print statements were initially used to make troubleshooting of the code easier, but they are now redundant. I will provide you with a code snippet to rewrite. After receiving the code snippet, rewrite it in a way that leaves the program unchanged but removes unnecessary print statements. If you understand, say "Understood" and I will proceed by providing you with a code snippet. Repeat this process each time I provide you with a new code snippet. 

  

explain 0: goal(remove all unnecessary prints from all code snippets I will provide you with and write the updated code snippets) 

explain 1: the prints were used to make troubleshooting of the code easier and are redundant now 

input 0: the code snippet to rewrite 

do 0: after input 0, rewrite the code snippet from input 0 in a way that leaves the program unchanged, but removes unnecessary prints 

output 0: "Understood." + continue with input 0 every time the user provides a code snippet 

  

 

 


# Iterative Improvement: 

  

I modified the prompts to require a specific minimum number of questions and to provide at least a certain number of reasons for something. Additionally, I removed some redundant parts of the guide. In the original guide and prompts, the number of questions was determined by the model, and it often didn't ask enough questions. This resulted in a lengthy and tedious troubleshooting process, even with the troubleshooting prompt. The reason for this is that the model was often uncertain about what caused the error and just tried to solve one of the possible issues without questioning whether this really is the correct cause of the problem it needs to solve. Now, I made the model provide at least three reasons for the issue, along with the likelihood of each reason being the actual cause (ranked on a scale of 1-10). The model then had to ask at least five questions to help identify the real issue. I also made the model print every variable, its value, type, and shape (if applicable). This significantly sped up troubleshooting, and where the model previously struggled to find the reason for errors, it identified them quite accurately now. 

  

The only challenge was the model's coding capability. The model is good at helping people by providing them with near-working or working code that a human programmer could finish and tweak, saving the programmer a lot of time. If the program to be coded isn't too complex (even if the amount of code is large), the model can do that as well. However, as of now, it struggles to create optimized code or code beyond a certain level of complexity. While it is quite impressive that GPT-4 was able to code a working Transformer model that took in English sentences and outputted German words, the Transformer's accuracy was quite low. Probable causes of the Transformer's low performance are its small size and having been trained for only 10 epochs, which can be easily rectified by increasing the values of training hyperparameters. Optimizing the Transformer would be difficult to rectify without programming experience. 

Overall, breaking the problem down into smaller parts and providing the model with enough data improved the model's ability to code by itself. Most importantly, it was much easier and smoother for me to repeatedly use the same prompts and answer a few simple questions than to try to come up with prompts on the go. An easy way to increase the model's capabilities would be to give it access to data on the internet, as is the case with projects like AutoGPT, or to connect it with neural networks trained to do various different tasks, like the project Hugging GPT. However, even with the standard model, users can still achieve impressive results by leveraging the model's capabilities and cleverly overcoming its weaknesses. 

 

 
 

# Conclusion: 

  

Throughout this project, I aimed to create a guide for using GPT-4 for coding even with minimal programming experience. I employed refined prompts that transformed GPT-4 into a clever coding assistant. I faced several challenges while working on this project, but ultimately, I completed my guide and prompts, which you can check by reading the file **"Guide.md"** that is located in this repository. This file also contains some common issues I encountered during my time working with GPT-4 and how to solve them. I allowed ChatGPT to code all my code, providing only vague suggestions in my answers (as if I had little programming experience and no machine learning experience). The model can be used by non-programmers to write code that isn't overly complex and by programmers to code large portions of the code, thus saving time for the programmer, as they only need to tweak and upgrade the code after ChatGPT quickly writes a near-working solution. This project aimed to create general prompts for coding any type of code. In my future projects, I plan to test the model's coding capabilities after providing it with prompts containing specific domain knowledge needed for effectively writing the code (e.g., specific prompts used for coding a convolutional neural network). 

  

By utilizing clever prompting architectures, this project demonstrated the potential of GPT-4 as a coding assistant that can save time for both non-programmers and experienced programmers. With further refinement and domain-specific prompts, GPT-4's coding capabilities could be expanded and improved. The success of this project highlights the importance of effectively working with large language models and refining prompt engineering techniques to overcome their limitations and harness their potential. 

  

In conclusion, this project serves as a foundation for my future exploration and research into the capabilities of GPT-4 and other large language models for coding tasks. By iteratively improving prompts and leveraging domain-specific knowledge, it may be possible to further enhance the efficiency and effectiveness of these models in various coding applications, contributing to the broader fields of artificial intelligence, and programming. 
