# Originally created June 2018 by Tyler Sorenson
# Formats the scrubbed text file

open my $input, '<', "$ARGV[0]" or die "Cannot open file: $!";
$rand_int = int(rand(10000000000000000));
$temp_filename = "updatedFile" . "$rand_int" . "txt";
open my $output, '>', $temp_filename or die "Cannot open file: $!";

while ( <$input> ) {
    if (m/\|\|\|\|END_OF_RECORD/g) {
        last;
    }
    s/[1-9]* ?\(?(un|ambig|popular\/ambig|[1-9]|10|11|NI|LF|PTilte|[2,4] digits|PRE|NameIs|Universities|Prefixes|STitle|Titles|Titles ambig|NamePattern[0-6]*|MD)?\)?\*\*]/\*\*]/g;

    if ($. > 2) {
        print $output $_;
    }
}

close $input;
close $output;

unlink("$ARGV[0]");
#rename "updatedFile.txt", "$ARGV[0]_clean.txt";
rename $temp_filename, "$ARGV[0]";
