document.addEventListener("DOMContentLoaded", function() {
    
    const sampleCaptions = [
        
        "Caption for Element 1",

    ];

    const captionsContainer=document.getElementById("captionsContainer");

    sampleCaptions.forEach((caption, index)=>{

        const captionElement = document.createElement("div");
        captionElement.classList.add("caption");
        captionElement.textContent = `Caption for image ${index + 1}: ${caption}`;
        captionsContainer.appendChild(captionElement);

    });
});
