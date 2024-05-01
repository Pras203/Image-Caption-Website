
let signupBtn = document.getElementById("signupBtn");
let signinBtn = document.getElementById("signinBtn");
let nameField = document.getElementById("nameField");
let confirmPassword=document.getElementById("confirmPassword");
let emailField = document.querySelector('input[type="email"]');
let passwordField = document.querySelector('input[type="password"]');
let title = document.getElementById("title");
let x=1;
let x1=0;
signinBtn.onclick=function(){

    nameField.style.maxHeight="0";
    confirmPassword.style.maxHeight="0";
    title.innerHTML="Log In";
    signupBtn.classList.add("disabled");
    signinBtn.classList.remove("disabled");

}
signinBtn.ondblclick=function(){
    validateSignupForm(x1);
    
}
signupBtn.onclick = function () {
    nameField.style.maxHeight = "60px";
    confirmPassword.style.maxHeight="60px";
    title.innerHTML = "Sign Up";
    signupBtn.classList.remove("disabled");
    signinBtn.classList.add("disabled");

    
};

signupBtn.ondblclick=function(){
    validateSignupForm(x);
}
function validateSignupForm(x) {
  
    let isValid = true;
    if(x){
    let username = nameField.querySelector('input[type="text"]').value.trim();
    if (username === ""||!/^[a-zA-Z]/.test(username)) {
        
        alerta("Please enter a username");
        isValid = false;
    }
    }

 
    let email = emailField.value.trim();
    if (email === "" || !isValidEmail(email)) {
        
        alerta("Please enter a valid email address");
        isValid = false;
    }

    let password = passwordField.value.trim();
    if (password === ""||!checkPassword(password)) {
        
        alerta("Please enter a password");
        isValid = false;
    }

    return isValid;
}
function checkPassword(str)
{
    var re = /^(?=.*\d)(?=.*[!@#$%^&*])(?=.*[a-z])(?=.*[A-Z]).{8,}$/;
    return re.test(str);
}

function isValidEmail(email) {
   
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}
function alerta(message) {
    // Display the alert after a delay of 1000 milliseconds (1 second)
    window.alert(message);
    setTimeout(function () {
        // Close the alert after 3 seconds
        document.querySelector(".alert").style.display = "none";
    }, 30);
}