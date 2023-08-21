<?php
/*
 @Author: Sanjay Kumar
 @Project: PHP MySQLi File Upload with Progress Bar Using jQuery Ajax
 @Email: onlinewebtutorhub@gmail.com
 @Website: https://onlinewebtutorblog.com/
*/

// Database configuration
$host   = "localhost";
$dbuser = "root";
$dbpass = "";
$dbname = "myarticlrs";

// Create database connection
$conn = new mysqli($host, $dbuser, $dbpass, $dbname);

// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}
