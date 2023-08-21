/*
 @Author: Sanjay Kumar
 @Project: PHP MySQLi File Upload with Progress Bar Using jQuery Ajax
 @Email: onlinewebtutorhub@gmail.com
 @Website: https://onlinewebtutorblog.com/
*/

$(document).ready(function() {

    $('#submitButton').click(function() {
        $('#uploadForm').ajaxForm({
            target: '#outputImage',
            url: 'process.php',
            beforeSubmit: function() {
                $("#outputImage").hide();
                if ($("#uploadImage").val() == "") {
                    $("#outputImage").show();
                    $("#outputImage").html("<div class='alert alert-danger'>Choose a file to upload.</div>");
                    return false;
                }

                $("#progressDivId").css("display", "block");

                var percentValue = '0%';

                $('#progressBar').width(percentValue);
                $('#percent').html(percentValue);

            },

            uploadProgress: function(event, position, total, percentComplete) {

                var percentValue = percentComplete + '%';
                $("#progressBar").animate({
                    width: '' + percentValue + ''
                }, {
                    duration: 5000,
                    easing: "linear",
                    step: function(x) {
                        percentText = Math.round(x * 100 / percentComplete);
                        $("#percent").text(percentText + "%");
                        if (percentText == "100") {
                            $("#outputImage").show();
                            $("#error").html("<div class='alert alert-success mt-2'>Image Successfully Uploaded.</div>");

                        }
                    }
                });
            },
            error: function(response, status, e) {
                alert('Oops something went.');
            },

            complete: function(xhr) {
                if (xhr.responseText && xhr.responseText != "error") {
                    $("#outputImage").html(xhr.responseText);
                } else {
                    $("#outputImage").show();
                    $("#outputImage").html("<div class='alert alert-danger'>Problem in uploading file.</div>");
                    $("#progressBar").stop();
                }
            }
        });
    });
});
