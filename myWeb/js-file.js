function ask() {   
    var data = document.getElementById("question").value 
    eel.submit(data)(call_Back)  
}  
  
function call_Back(output){  
    document.getElementById("response").value = output  
} 