# Pytorch-Project
Template to carry out a pytorch project

The template consist of a log which builds with GitHub workflows. It's written in markdown and will be linked to the repo
page which is pretty useful to share information.  


There is  main_library that can be packaged and exported easily. The main library contains a `core` that uses minimal 
libraries to run (this contains your model and the basic things you need it to run).  

There are other sublibraries which can hold more dependancies but, in practice, they don't need to be loaded. Maybe for 
training, for evaluation... 

