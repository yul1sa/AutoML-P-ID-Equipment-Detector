# P&ID Equipment Data Chatbot Web Application

## Description

This project is a web-based application that allows users to enter equipment data associated with a P&ID (Piping and Instrumentation Diagram) document and then query this data using a simple chatbot interface. It is an adaptation of an original console-based Python script into a Flask web application.

The application first prompts for a P&ID document name. Then, users can input details for multiple pieces of equipment, including type, tag ID, detection confidence, bounding box coordinates, and notes. Once data is entered, users can navigate to a chat interface to ask questions about the entered equipment.

## Features

* Web-based interface for P&ID document name entry.
* Web form for entering multiple equipment items with details.
* Dynamic display of currently entered equipment.
* Chat interface to query the entered equipment data.
* Ability to reset all entered data and start over.
