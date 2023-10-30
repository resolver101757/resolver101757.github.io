# commands to run quarto

to create a project (run once at the beginning to create the project files)
- quarto create project_name --template=blog

to run a preview locally
- quarto preview .

to Create update the files (run before pushing to github)
- quarto render .

# links :

- Video of somone showing how to use quarto https://www.youtube.com/watch?v=nllKcuX7rEc
- ilovetensoor coppied a good link here : https://discord.com/channels/689892369998676007/1096981866248147014/1167453441291984936


# quarto reference page : 

https://quarto.org/docs/reference/cells/cells-jupyter.html


# setup for using colab with quarto

- Create a new notebook in colab and add markup in markup - no need for RAW as documentation says.
- install the drive app in windows which will reveal the drive folder in the file explorer, i used streams so no files are stored locally.
- Create a script to copy file to local installation and run "quarto render ." to create the html file. see file located C:\development\github projects\AlexPaulKelly\copy_lsuv_from_colab.bat
- Edit the notebook in colab and run the script to render the html file, repeat until happy with the result and publish to github.




