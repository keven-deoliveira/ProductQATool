# install eel
# pip install eel

# importing the eel library  
import eel
import ask_openai

# initializing the application  
eel.init("../myWeb")


@eel.expose
def submit(data):
    try:
        return ask_openai.ask_openai(data)
    except KeyboardInterrupt:
        pass


# starting the application
eel.start("webApp.html", size=(900, 550))
