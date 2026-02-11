// Module for a 2-to-1 Multiplexer
module mux_2_to_1 (
    input  wire a,    // Input 0
    input  wire b,    // Input 1
    input  wire sel,  // Select signal
    output wire out   // Output
);

    assign out = sel ? b : a; // Conditional assignment based on 'sel'

endmodule
