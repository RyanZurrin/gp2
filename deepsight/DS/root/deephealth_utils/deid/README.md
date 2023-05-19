# deid
Scripts for removing PHI from free text  

## pythonWrapper.py
Contains the funtion scrub_wrapper, which calls the perls scripts which do the actual scrubbing  
scrub_wrapper takes 2 arguments:  
1. Existing text file to scrub
2. Scrubbed file to create

## phi_tools.py
Contains functions that perform various tasks related to phi  

### contains_phi(input)
Checks if there is phi in a text file or string, and returns True if there is phi, False otherwise   
1. input: Path to text file to check, or string to check

### replace_phi(input)
Takes in a string of free text, and returns a scrubbed version of the string  
1. input: String to scrub

## addHeader.pl
Adds a header to the text file so that the deid script processes the text properly  
Input:  
1. File to scrub
2. File to scrub (with header added)

Output:  
1. File to scrub (with header added)

## deid.pl
Scrubs the text and outputs it as a new file, deletes the input file with the header added  

Input:  
1. File to scrub (with header added)
2. Scrubbed file
3. Path to deid-output.config
4. Path to deid directory
5. Option to print commentary to terminal, 'COMMENTATE' will yield output

Output:  
1. Scrubbed file

## scrubFormatter.pl
Formats the scrubbed file  
Input:  
1. Scrubbed file

Output:  
1. Formatted scrubbed file
