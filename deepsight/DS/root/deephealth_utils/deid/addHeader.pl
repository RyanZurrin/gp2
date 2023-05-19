# Originally created June 2018 by Tyler Sorenson
# Copies the input file to the output file along with a header and footer.

open my $input, '<', "$ARGV[0]" or die "Cannot open file: $!";
open my $output, '>', "$ARGV[1]" or die "Cannot open file: $!";

print $output "START_OF_RECORD=1||||1||||\n";

while (<$input>) {
    print $output $_;
}
print $output "\n||||END_OF_RECORD";
