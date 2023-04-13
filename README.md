# Product Q&A Tool

A Product Q&A system created for company Hackathon competition. The system is meant to supplement or even replace traditional product help guide, acting as a kind of "knowledgeable assistant" that can answer questions about the product. We used existing examples as guidelines for this project.

The system uses GPT3 as an embedding model, which translates the question into a vector to represent the meaning behind the query. This vector can be passed through a knowledgebase, which can be loaded into a vector database e.g. Pinecone to keep information encoded and indexed. For the sake of time, we stored our data locally. A search is performed against the vector database using the translated vector to return the k-nearest vectors that would generate the most accurate and appropriate answer. The result is translated back to text using the text-davinci-003 model.
 
We created this system in order to simplify the user experience of accessing knowledgebases. Traditional keyword-based searching can be time-consuming and frustrating for a user, especially if they are new to the product and do not know the right keywords for product terminology. Our product is known to be complex to configure. By providing a Q&A system to act as an assistant, users can ask questions directly and cut out the tedious process of manually analyzing documents for answers. This system could also be used as a learning tool, as it can clarify terminology, explain concepts, and list any steps needed to configure specific aspects of the product.
 
Originally, we wanted to implement voice input and output capabilities to the project. Ideally, a user could verbally ask a question to the Q&A tool and get a text or voice response. This could prove useful in scenarios where a user needs an answer on demand as they are configuring some specific use case or troubleshooting. Voice capabilities also allows the project to be more accessible. In addition, the current version of the POC is loaded with a very small knowledgebase compared to the real product knowledgebase. Ideally the AI will be able to handle very large knowledgebase and the entire product guide would be accessible by the AI to provide the best possible answers.

Since this project was created for company usage, sensitive material was removed from the source files. Due to this, running the project as it is on Github will not work.

______

# Examples

![example1](https://github.com/keven-deoliveira/ProductQATool/blob/main/images/example1.png)

![example2](https://github.com/keven-deoliveira/ProductQATool/blob/main/images/example2.png)

![example3](https://github.com/keven-deoliveira/ProductQATool/blob/main/images/example3.png)

![example4](https://github.com/keven-deoliveira/ProductQATool/blob/main/images/example4.png)
