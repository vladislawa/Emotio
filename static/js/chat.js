function getBotResponse() {
    
    var rawText = $("#textInput").val();
    var userHtml = '<p class="userText"><span>' + rawText + "</span></p>";
    $("#textInput").empty();
    $("#chatbox").empty();
    $("#textInput").val("");
    $("#chatbox").append(userHtml);
    document
      .getElementById("userInput")
      .scrollIntoView({ block: "start", behavior: "smooth" });

    if (document.getElementById("userInput") == "Sing a song") {
       var audio = new Audio('../static/twinkle.mp3');
       audio.play();
    } 

    $.get("/get", { msg: rawText }).done(function(data) {
      var botHtml = '<p class="botText"><span>' + data + "</span></p>";
      $("#chatbox").append(botHtml);
      document
        .getElementById("userInput")
        .scrollIntoView({ block: "start", behavior: "smooth" });
    });
  }
  $("#textInput").keypress(function(e) {
    if (e.which == 13) {
      getBotResponse();
    }
  });
