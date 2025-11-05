import { View, Text, StyleSheet, Button } from 'react-native';
import { router } from 'expo-router';

export default function HomeScreen() {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Slope Mapping App</Text>
      <Text style={styles.subtitle}>Analyze golf green elevation & breaks</Text>

      <Button 
        title="View Course Maps" 
        onPress={() => {}} 
      />

      <Button 
        title="View Green Maps" 
        onPress={() => {}}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  title: {
    fontSize: 28,
    fontWeight: '600',
    marginBottom: 10,
  },
  subtitle: {
    fontSize: 16,
    marginBottom: 30,
    textAlign: 'center',
  },
});
